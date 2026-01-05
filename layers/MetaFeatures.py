import torch
import torch.nn.functional as F

def compute_meta_features(x):
    """
    Function 1: Compute multiple meta-features for time series on GPU.
    
    Args:
        x: Input tensor of shape (B, L, D) (Batch, Length, Dimension).
           B: Batch size
           L: Time steps
           D: Number of variables
           
    Returns:
        meta_features: Tensor of shape (B, D, N) (Batch, Dimension, Num_Features).
        Features are:
        0: Periodicity Strength (Top-1 Power Ratio)
        1: Trend Strength (Absolute Correlation with Time)
        2: Spectral Entropy (Shannon Entropy of Power Spectrum)
        3: Volatility/Complexity (Ratio of First Diff Std to Raw Std)
    """
    B, L, D = x.shape
    device = x.device
    eps = 1e-8

    # Ensure x is float
    x = x.float()

    # 1. Periodicity Strength using FFT
    # Shape: (B, L//2 + 1, D) after rfft
    fft = torch.fft.rfft(x, dim=1)
    power_spectrum = torch.abs(fft)**2
    
    # Exclude DC component (0-th frequency)
    valid_power = power_spectrum[:, 1:, :]  # (B, L//2, D)
    
    sum_power = torch.sum(valid_power, dim=1) + eps
    max_power = torch.max(valid_power, dim=1).values
    
    # Feature 0: Periodicity
    periodicity = max_power / sum_power # (B, D)

    # 2. Trend Strength (Correlation with Time)
    # Create time indices
    t = torch.arange(L, device=device, dtype=torch.float32).view(1, L, 1) # (1, L, 1)
    t = t.expand(B, L, D)
    
    # Center data
    x_mean = torch.mean(x, dim=1, keepdim=True)
    t_mean = torch.mean(t, dim=1, keepdim=True)
    
    x_centered = x - x_mean
    t_centered = t - t_mean
    
    covariance = torch.sum(x_centered * t_centered, dim=1)
    std_x = torch.sqrt(torch.sum(x_centered**2, dim=1))
    std_t = torch.sqrt(torch.sum(t_centered**2, dim=1))
    
    # Feature 1: Trend (Absolute Correlation)
    trend = torch.abs(covariance / (std_x * std_t + eps)) # (B, D)

    # 3. Spectral Entropy
    # Normalize power spectrum to get probability distribution
    psd_norm = valid_power / sum_power.unsqueeze(1) # (B, L//2, D)
    entropy = -torch.sum(psd_norm * torch.log(psd_norm + eps), dim=1) # (B, D)
    
    # Feature 2: Spectral Entropy (Raw value, will be normalized later)
    spectral_entropy = entropy

    # 4. Volatility / Complexity (Mobility approximation)
    # Std of first difference / Std of signal
    diff_x = x[:, 1:, :] - x[:, :-1, :]
    std_diff = torch.std(diff_x, dim=1)
    std_raw = torch.std(x, dim=1) + eps
    
    # Feature 3: Volatility
    volatility = std_diff / std_raw # (B, D)

    # Stack all features: (B, D, 4)
    meta_features = torch.stack([periodicity, trend, spectral_entropy, volatility], dim=-1)
    
    return meta_features


def normalize_meta_features(meta_features, L):
    """
    Function 2: Normalize meta-features based on theoretical bounds.
    
    Args:
        meta_features: Tensor of shape (B, D, N) from compute_meta_features.
        L: Length of original time series (needed for Entropy bound).
        
    Returns:
        normalized_features: Tensor of shape (B, D, N) in range [0, 1].
    """
    # Clone to avoid in-place modification of input
    norm_feats = meta_features.clone()
    eps = 1e-8
    
    # Feature 0: Periodicity (Already 0-1 ratio)
    norm_feats[..., 0] = torch.clamp(norm_feats[..., 0], 0, 1)
    
    # Feature 1: Trend (Correlation is -1 to 1, we took Abs, so 0-1)
    norm_feats[..., 1] = torch.clamp(norm_feats[..., 1], 0, 1)
    
    # Feature 2: Spectral Entropy
    # Max entropy for N bins is log(N). Here bins = L//2
    max_entropy = torch.log(torch.tensor(L // 2, dtype=torch.float32, device=meta_features.device))
    norm_feats[..., 2] = norm_feats[..., 2] / (max_entropy + eps)
    norm_feats[..., 2] = torch.clamp(norm_feats[..., 2], 0, 1)
    
    # Feature 3: Volatility
    # Theoretically can be > 1. Empirically usually < 2 for smooth signals.
    # We use a sigmoid-like squashing or simple clipping for safety.
    # Let's use Tanh to squash to 0-1 softly, assuming typical range is 0-2
    # Or just min-max if we knew bounds. Mobility of White Noise is ~ sqrt(2) ~ 1.41.
    # Mobility of Brown Noise is low.
    # Let's normalize by 2.0 and clip.
    norm_feats[..., 3] = torch.clamp(norm_feats[..., 3] / 2.0, 0, 1)
    
    return norm_feats


def select_visualization_method(meta_features):
    """
    Function 3: Select suitable visualization method based on meta-features.
    
    Args:
        meta_features: Tensor of shape (B, D, N) (Raw or Normalized - Code assumes consistent scale).
                       We expect UNNORMALIZED input as per prompt requirement for Function 3 input,
                       but we will normalize internally or use robust thresholds.
                       The prompt says: "Input is unnormalized raw meta features".
                       
    Returns:
        selection_indices: Tensor of shape (B, D) containing indices of selected methods.
        ts2img_methods = ["wavelet", "mel", "mtf", "seg", "gaf", "rp", "stft"]
        Indices:
        0: wavelet
        1: mel
        2: mtf
        3: seg
        4: gaf
        5: rp
        6: stft
    """
    # Methods map
    # 0: wavelet - Good for non-stationary, multi-scale
    # 1: mel     - Good for audio-like, high frequency complexity
    # 2: mtf     - Good for statistical transitions, discrete-like
    # 3: seg     - Good for periodic data
    # 4: gaf     - Good for trend/temporal correlation preservation
    # 5: rp      - Good for chaos/non-linearity/recurrence
    # 6: stft    - Good for strong time-frequency patterns
    
    # We define a score for each method based on features.
    # Features: [Periodicity, Trend, Entropy, Volatility]
    
    # Extract features for clarity
    periodicity = meta_features[..., 0]
    trend = meta_features[..., 1]
    entropy = meta_features[..., 2] # Note: This is raw entropy
    volatility = meta_features[..., 3]
    
    # We need to roughly normalize entropy for scoring since it depends on L
    # We can't know L here easily without passing it, but we can assume L is large enough or rely on relative values.
    # However, to be safe, let's use the features as is with weights.
    # Or better, let's just implement a robust decision logic (soft scoring).
    
    # Initialize scores (B, D, 7)
    scores = torch.zeros(meta_features.shape[:-1] + (7,), device=meta_features.device)
    
    # --- Scoring Logic ---
    
    # 1. SEG (Idx 3): Strongly prefers Periodicity
    scores[..., 3] = periodicity * 3.0
    
    # 2. GAF (Idx 4): Strongly prefers Trend (and Stationarity usually, but GAF handles non-stationarity well too by preserving structure)
    # Actually GAF is often used to encode static features or trends.
    scores[..., 4] = trend * 2.5
    
    # 3. STFT (Idx 6) & MEL (Idx 1): Prefer High Entropy (Rich frequency content)
    # We weight them similarly, maybe STFT slightly more for general purpose
    # Entropy is usually 0 ~ 5+.
    scores[..., 6] = entropy * 0.5
    scores[..., 1] = entropy * 0.45 
    
    # 4. RP (Idx 5): Recurrence Plot. Good for chaotic, non-linear, low trend but structured.
    # High complexity but maybe low periodicity?
    # Let's say RP scores high if volatility is high but it's not just pure noise (high entropy).
    # This is hard to heuristicize perfectly. 
    # Let's assume RP is good when Trend is Low and Periodicity is Low (Chaos).
    scores[..., 5] = (1.0 - periodicity) * (1.0 - trend) * 2.0
    
    # 5. MTF (Idx 2): Markov Transition Field. Good for volatility and statistical distribution.
    scores[..., 2] = volatility * 1.5
    
    # 6. Wavelet (Idx 0): Good for Non-stationary (changing stats).
    # High Volatility + High Entropy usually suggests Wavelet might be good.
    scores[..., 0] = (volatility + entropy * 0.2) * 1.0

    # Select best method
    selection_indices = torch.argmax(scores, dim=-1)
    
    return selection_indices
