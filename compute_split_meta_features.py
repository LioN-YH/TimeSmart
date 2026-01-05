import argparse
import json
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from types import SimpleNamespace

from data_provider.data_factory import data_provider
from layers.meta_feature_v import batch_extract_meta_features_gpu

def normalize_input(x, args_ns):
    means = x.mean(1, keepdim=True).detach()
    x = x - means
    # unbiased=False: use biased estimator (sample variance)
    # 1e-5: prevent division by zero
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    stdev /= args_ns.norm_const
    x = x / stdev
    return x, means, stdev

def build_args(dset, root_path, seq_len, label_len, pred_len, batch_size, num_workers):
    return SimpleNamespace(
        task_name="long_term_forecast",
        data=dset,
        root_path=root_path,
        data_path=f"{dset}.csv",
        features="M",
        target="OT",
        freq="h",
        embed="timeF",
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        seasonal_patterns=None,
        percent=1.0,
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation_ratio=0,
        norm_const=0.4,
    )

def compute_features(dset, args_ns, device, flag):
    _, loader = data_provider(args_ns, flag=flag)
    print(f"Processing {flag} set...")
    
    feats_list = []
    
    with torch.no_grad():
        for batch_x, _, _, _ in loader:
            x = batch_x.float().to(device)
            x, _, _ = normalize_input(x, args_ns)
            f = batch_extract_meta_features_gpu(x, args_ns.seq_len, args_ns.pred_len)
            feats_list.append(f.cpu().numpy())
            
    if not feats_list:
        return None
        
    # Concatenate all batches: (Total_Samples, N_vars, N_features)
    all_feats = np.concatenate(feats_list, axis=0)
    return all_feats

def plot_embedding(embeddings, labels, title, save_path):
    plt.figure(figsize=(10, 8))
    
    # labels: 0 for train, 1 for val, 2 for test
    colors = ['r', 'g', 'b']
    names = ['Train', 'Val', 'Test']
    
    for i in range(3):
        mask = labels == i
        if np.any(mask):
            plt.scatter(embeddings[mask, 0], embeddings[mask, 1], c=colors[i], label=names[i], alpha=0.6, s=10)
            
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Dataset name in data_provider dict keys")
    parser.add_argument("--dset", type=str, required=True, help="Dataset filename prefix (e.g. ETTh1)")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples_for_plot", type=int, default=2000, help="Max samples to use for plotting to save time")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ns = build_args(
        args.dset,
        "./dataset/",
        args.seq_len,
        48,
        args.pred_len,
        args.batch_size,
        4, # num_workers
    )
    ns.data = args.data
    
    os.makedirs(args.output, exist_ok=True)
    
    splits = ['train', 'val', 'test']
    split_data = {}
    stats = {}
    
    # 1. Compute and Save Features
    for i, split in enumerate(splits):
        feats = compute_features(args.dset, ns, device, split)
        if feats is not None:
            # feats shape: (Samples, N_vars, N_meta_feats)
            print(f"  {split} features shape: {feats.shape}")
            
            # Save .npy
            npy_path = os.path.join(args.output, f"{args.dset}_{split}_features.npy")
            np.save(npy_path, feats)
            print(f"  Saved features to {npy_path}")
            
            # Compute Mean/Std (over samples)
            mean_val = feats.mean(axis=0)
            std_val = feats.std(axis=0)
            
            stats[split] = {
                "mean": mean_val.tolist(),
                "std": std_val.tolist()
            }
            
            split_data[split] = feats
        else:
            print(f"Warning: No data found for {split}")
            
    # 2. Save Stats JSON
    json_path = os.path.join(args.output, f"{args.dset}_stats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
    print(f"Saved stats to {json_path}")
        
    # 3. Plotting
    if not split_data:
        print("No data to plot.")
        return

    # Check number of variables
    first_split = list(split_data.keys())[0]
    N_vars = split_data[first_split].shape[1]
    
    print(f"Plotting for {N_vars} variables...")
    
    for var_idx in range(N_vars):
        print(f"Processing Variable {var_idx}...")
        # Collect data for this variable across splits
        X_list = []
        labels_list = []
        
        for i, split in enumerate(splits):
            if split in split_data:
                data = split_data[split][:, var_idx, :] # (Samples, 15)
                
                # Subsample if necessary for plotting performance
                if data.shape[0] > args.max_samples_for_plot:
                    indices = np.random.choice(data.shape[0], args.max_samples_for_plot, replace=False)
                    data = data[indices]
                
                X_list.append(data)
                labels_list.append(np.full(data.shape[0], i))
        
        if not X_list:
            continue
            
        X = np.concatenate(X_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        # Normalize features before embedding? Usually good practice, but t-SNE handles scale reasonably well if uniform.
        # Let's standardize for better embedding results.
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X_norm = (X - X_mean) / X_std
        
        # Replace NaNs if any (should have been handled in extraction, but safety first)
        X_norm = np.nan_to_num(X_norm)

        # t-SNE
        print(f"  Running t-SNE...")
        perplexity = 30 if X_norm.shape[0] > 30 else max(1, X_norm.shape[0] // 2)
        try:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
            X_tsne = tsne.fit_transform(X_norm)
            plot_embedding(X_tsne, labels, f"t-SNE Variable {var_idx} ({args.dset})", 
                           os.path.join(args.output, f"{args.dset}_var{var_idx}_tsne.png"))
        except Exception as e:
            print(f"  t-SNE failed for variable {var_idx}: {e}")
        
        # UMAP
        print(f"  Running UMAP...")
        try:
            reducer = umap.UMAP(random_state=42, n_jobs=1) # n_jobs=1 to avoid some potential issues in restricted envs
            X_umap = reducer.fit_transform(X_norm)
            plot_embedding(X_umap, labels, f"UMAP Variable {var_idx} ({args.dset})", 
                           os.path.join(args.output, f"{args.dset}_var{var_idx}_umap.png"))
        except Exception as e:
            print(f"  UMAP failed for variable {var_idx}: {e}")

if __name__ == "__main__":
    main()
