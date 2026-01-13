import pandas as pd
import numpy as np

def get_periodicity_score(series):
    # Simple FFT-based periodicity score
    fft_vals = np.fft.rfft(series - np.mean(series))
    power = np.abs(fft_vals)**2
    # Ignore DC component
    power = power[1:]
    # Score is ratio of max power to total power (indicating concentration at a specific frequency)
    score = np.max(power) / (np.sum(power) + 1e-10)
    return score

# Load the dataset
df = pd.read_csv('dataset/ETTh1.csv')

# --- Truncation Logic ---
# Based on analysis, HUFL shows concept drift (starts positive, then shifts/widens range).
# We identify the point where the distribution characteristics change.
# A robust heuristic here is the first time HUFL drops below 0 (or close to it), 
# which marks the transition to the regime consistent with the latter half (test set).
# We search for the first index where HUFL < 0.
neg_hufl_indices = df.index[df['HUFL'] < 0]
if len(neg_hufl_indices) > 0:
    start_idx = neg_hufl_indices[0]
    # Optional: back off slightly to capture the transition, or just cut there.
    # We'll use this point as the start of the "new" consistent regime.
    print(f"Detected concept drift start at index {start_idx} (HUFL < 0).")
else:
    # Fallback if no negative values found (unlikely given analysis)
    start_idx = int(len(df) * 0.25) # Drop first 25%

# Ensure we keep "as much data as possible" while being consistent
# If the cut is too aggressive (e.g. > 50%), we might reconsider, but here it's ~27% cut.
df_truncated = df.iloc[start_idx:].reset_index(drop=True)

print(f"Original shape: {df.shape}")
print(f"Truncated shape: {df_truncated.shape}")
# ------------------------

# Exclude 'date' and 'OT' for feature selection
cols = [c for c in df_truncated.columns if c not in ['date', 'OT']]

scores = {}
for col in cols:
    scores[col] = get_periodicity_score(df_truncated[col].values)

# Select top 3 features
sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
top_features = [x[0] for x in sorted_features[:3]]

print("Top 3 periodic features:", top_features)

# Create optimized dataset: date, top_features, OT
out_cols = ['date'] + top_features + ['OT']
df_opt = df_truncated[out_cols]

# Save to new csv
output_path = 'dataset/ETTh1_opt.csv'
df_opt.to_csv(output_path, index=False)
print(f"Saved optimized dataset to {output_path}")
