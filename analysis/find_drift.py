import pandas as pd
print("Starting analysis...")
try:
    df = pd.read_csv('dataset/ETTh1.csv')
    print(f"Loaded df with shape {df.shape}")

    # Check when HUFL goes below 0 for the first time
    neg_hufl = df[df['HUFL'] < 0]
    if not neg_hufl.empty:
        first_neg = neg_hufl.index[0]
        print(f"First negative HUFL at index: {first_neg}")
    else:
        print("No negative HUFL found.")

    # Print segment stats
    segment_size = len(df) // 4
    for i in range(4):
        start = i * segment_size
        end = (i+1) * segment_size
        print(f"\nSegment {i} ({start}-{end}):")
        print(df.iloc[start:end][['HUFL', 'OT']].describe().to_string())
except Exception as e:
    print(f"Error: {e}")
