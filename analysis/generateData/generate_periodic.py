import numpy as np
import pandas as pd
import os


def generate_strong_periodic_series(
    n, base_period, amplitude=1.0, noise_std=0.05, phase=0.0
):
    """
    Generates a single sine wave with noise.
    """
    t = np.arange(n)
    signal = amplitude * np.sin(2 * np.pi * t / base_period + phase)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=n)
    return signal + noise


def generate_data():
    n = 512
    num_vars = 6
    output_dir = "/home/user10/TimeSmart/dataset"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "synthetic_periodic.csv")

    # Define different periods for the 6 variables
    # These periods are chosen to be distinct to satisfy the requirement
    periods = [24, 48, 12, 60, 96, 30]
    amplitudes = [1.0, 1.5, 0.8, 1.2, 0.5, 2.0]

    data = {}
    # Creating a date column as is common in time series datasets
    data["date"] = pd.date_range(start="2023-01-01", periods=n, freq="H")

    np.random.seed(42)

    for i in range(num_vars):
        col_name = f"var_{i+1}"
        period = periods[i]
        amp = amplitudes[i]
        phase = np.random.uniform(0, 2 * np.pi)

        # Generate the series
        series = generate_strong_periodic_series(n, period, amplitude=amp, phase=phase)
        data[col_name] = series

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated data saved to {output_path}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    generate_data()
