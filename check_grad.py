
import torch
import sys
import os

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

from src.TimeVLM.model_v_c import Model

class Config:
    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.is_training = 1
        self.model_id = 'test'
        self.model = 'TimeVLM_v_c'
        self.data = 'ETTh1'
        self.root_path = './dataset/'
        self.data_path = 'ETTh1.csv'
        self.features = 'M'
        self.target = 'OT'
        self.freq = 'h'
        self.checkpoints = './checkpoints/'
        self.seq_len = 512
        self.label_len = 48
        self.pred_len = 96
        self.seasonal_patterns = 'Monthly'
        self.inverse = False
        self.mask_rate = 0.25
        self.anomaly_ratio = 0.25
        self.expand = 2
        self.d_conv = 4
        self.top_k = 5
        self.num_kernels = 6
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 128
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 768
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.1
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.channel_independence = 1
        self.decomp_method = 'moving_avg'
        self.use_norm = 1
        self.down_sampling_layers = 0
        self.down_sampling_window = 1
        self.down_sampling_method = None
        self.seg_len = 48
        self.num_workers = 10
        self.itr = 1
        self.train_epochs = 10
        self.batch_size = 32
        self.patience = 5
        self.learning_rate = 0.001
        self.des = 'Exp'
        self.loss = 'MSE'
        self.lradj = 'type1'
        self.use_amp = False
        self.vlm_type = 'clip'
        self.image_size = 56
        self.memory_bank_size = 20
        self.patch_memory_size = 100
        self.periodicity = 24
        self.interpolation = 'bilinear'
        self.norm_const = 0.4
        self.three_channel_image = True
        self.finetune_vlm = False
        self.learnable_image = True
        self.save_images = False
        self.use_cross_attention = True
        self.w_out_visual = False
        self.w_out_text = False
        self.w_out_query = False
        self.visualize_embeddings = False
        self.use_mem_gate = False
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'

def check_gradients():
    config = Config()
    
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
        config.use_gpu = False
        config.gpu = None

    print("Initializing model...")
    model = Model(config).to(device)
    
    # Generate dummy data
    B = 2
    L = config.seq_len
    D = config.enc_in
    x_enc = torch.randn(B, L, D).to(device)
    
    print("Running forward pass...")
    # Forward pass
    outputs = model(x_enc)
    
    # Calculate dummy loss
    loss = outputs.mean()
    
    print("Running backward pass...")
    # Backward pass
    loss.backward()
    
    print("\n--- Gradient Check Results ---")
    
    # Check gradients for LearnableTimeSeriesToImage_v
    print("\nChecking LearnableTimeSeriesToImage_v gradients:")
    has_grad_image = False
    for name, param in model.learnable_image_module.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            print(f"  {name}: grad_mean={grad_mean:.6f}, grad_max={grad_max:.6f}")
            if grad_max > 0:
                has_grad_image = True
        else:
            print(f"  {name}: grad is None")
            
    if not has_grad_image:
        print("🔴 CRITICAL: No gradients found in LearnableTimeSeriesToImage_v module! (Gradient Cutoff Confirmed)")
    else:
        print("🟢 LearnableTimeSeriesToImage_v is receiving gradients.")

    # Check gradients for Prediction Head
    print("\nChecking Prediction Head gradients:")
    has_grad_head = False
    for name, param in model.prediction_head.named_parameters():
        if param.grad is not None:
            if param.grad.abs().max().item() > 0:
                has_grad_head = True
            # print(f"  {name}: grad exists")
        else:
            print(f"  {name}: grad is None")
            
    if has_grad_head:
        print("🟢 Prediction Head is receiving gradients (Expected).")
    else:
        print("🔴 Prediction Head is NOT receiving gradients.")

if __name__ == "__main__":
    try:
        check_gradients()
    except Exception as e:
        import traceback
        traceback.print_exc()
