import numpy as np
import pandas as pd
import os


def generate_linear_trend(n, slope, intercept, noise_std=1.0):
    t = np.arange(n)
    signal = slope * t + intercept
    noise = np.random.normal(loc=0.0, scale=noise_std, size=n)
    return signal + noise


def generate_quadratic_trend(n, a, b, c, noise_std=1.0):
    t = np.arange(n)
    signal = a * t**2 + b * t + c
    noise = np.random.normal(loc=0.0, scale=noise_std, size=n)
    return signal + noise


def generate_exponential_trend(n, base, scale, shift, noise_std=1.0):
    t = np.arange(n)
    # Using a normalized time t_norm from 0 to 10 for exponential stability
    t_norm = np.linspace(0, 5, n)
    signal = scale * np.exp(base * t_norm) + shift
    noise = np.random.normal(loc=0.0, scale=noise_std, size=n)
    return signal + noise


def generate_data():
    n = 512
    num_vars = 6
    output_dir = "/home/user10/TimeSmart/dataset"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "synthetic_trend_512.csv")

    data = {}
    # Creating a date column
    data["date"] = pd.date_range(start="2023-01-01", periods=n, freq="H")

    np.random.seed(123)  # Different seed from periodic

    # Variable 1: Strong Positive Linear Trend
    data["var_1"] = generate_linear_trend(n, slope=0.5, intercept=10, noise_std=2.0)

    # Variable 2: Strong Negative Linear Trend
    data["var_2"] = generate_linear_trend(n, slope=-0.3, intercept=200, noise_std=2.0)

    # Variable 3: Quadratic Trend (Accelerating upwards)
    # 512^2 is large, so coefficient must be small
    data["var_3"] = generate_quadratic_trend(n, a=0.002, b=0.1, c=50, noise_std=2.0)

    # Variable 4: Quadratic Trend (Concave down / slowing down)
    data["var_4"] = generate_quadratic_trend(n, a=-0.001, b=0.8, c=100, noise_std=2.0)

    # Variable 5: Exponential Trend
    data["var_5"] = generate_exponential_trend(
        n, base=0.8, scale=10, shift=0, noise_std=2.0
    )

    # Variable 6: Steeper Linear Trend
    data["var_6"] = generate_linear_trend(n, slope=1.2, intercept=-50, noise_std=5.0)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated data saved to {output_path}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    generate_data()
