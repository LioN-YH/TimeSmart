import numpy as np
import pandas as pd


def generate_strong_periodic_series(
    n, base_period, amplitude=1.0, noise_std=0.05, phase=0.0
):
    t = np.arange(n)
    signal = amplitude * np.sin(2 * np.pi * t / base_period + phase)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=n)
    return signal + noise


def generate_multi_periodic_series(
    n, base_periods, amplitudes, noise_std=0.05, phases=None
):
    t = np.arange(n)
    signal = np.zeros(n)
    if phases is None:
        phases = np.random.uniform(0, 2 * np.pi, len(base_periods))
    for period, amplitude, phase in zip(base_periods, amplitudes, phases):
        signal += amplitude * np.sin(2 * np.pi * t / period + phase)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=n)
    return signal + noise


def sawtooth_wave(n, period, phase=0.0):
    t = np.arange(n)
    r = (t / period + phase / (2 * np.pi)) % 1.0
    return 2.0 * r - 1.0


def square_wave(n, period, phase=0.0):
    t = np.arange(n)
    return np.where(np.sin(2 * np.pi * t / period + phase) >= 0.0, 1.0, -1.0)


def synthesize_etth_like(input_path, output_path, random_seed=42):
    np.random.seed(random_seed)
    df = pd.read_csv(input_path)
    n = len(df)

    result = pd.DataFrame()
    if "date" in df.columns:
        result["date"] = df["date"]
    else:
        result["date"] = np.arange(n)

    numeric_cols = [c for c in df.columns if c != "date"]
    amplitudes = [1.0, 1.2, 0.8, 1.5, 0.6, 1.8, 1.1]

    base_period_pool = [24, 24 * 7, 24 * 30, 24 * 90, 24 * 180, 24 * 365]
    complexity_by_col = {
        "HUFL": 6,
        "MUFL": 4,
        "LUFL": 2,
        "HULL": 3,
        "MULL": 5,
        "LULL": 3,
        "OT": 5,
    }
    default_complexity = 2

    for i, col in enumerate(numeric_cols):
        base_amplitude = amplitudes[i % len(amplitudes)]
        num_components = complexity_by_col.get(col, default_complexity)
        component_periods = base_period_pool[:num_components]
        amplitude_ratios = [1.0, 0.6, 0.3, 0.15, 0.07, 0.03]
        component_amplitudes = [
            base_amplitude * amplitude_ratios[j] for j in range(num_components)
        ]
        phases = np.random.uniform(0, 2 * np.pi, len(component_periods))
        noise_std = 0.05 * base_amplitude
        series = generate_multi_periodic_series(
            n=n,
            base_periods=component_periods,
            amplitudes=component_amplitudes,
            noise_std=noise_std,
            phases=phases,
        )

        if col in {"HUFL", "MUFL", "LUFL"}:
            phase_ns = np.random.uniform(0, 2 * np.pi)
            s = sawtooth_wave(n, 24, phase_ns)
            series += 0.5 * base_amplitude * s
        if col in {"OT"}:
            phase_ns = np.random.uniform(0, 2 * np.pi)
            q = square_wave(n, 24 * 7, phase_ns)
            series += 0.3 * base_amplitude * q

        if df[col].std() != 0 and not np.isnan(df[col].std()):
            series = (series - series.mean()) / (series.std() + 1e-8)
            series = series * df[col].std() + df[col].mean()

        result[col] = series

    result.to_csv(output_path, index=False)


if __name__ == "__main__":
    input_csv = "/home/user10/TimeSmart/dataset/ETTh1.csv"
    output_csv = "/home/user10/TimeSmart/dataset/ETTh1_synthetic_periodic.csv"
    synthesize_etth_like(input_csv, output_csv)
