import pandas as pd
from statsmodels.tsa.seasonal import STL
from pathlib import Path


def decompose_etth1(
    csv_path: Path,
    trend_path: Path,
    seasonal_path: Path,
    resid_path: Path,
    period: int = 24,
    seasonal: int = 13,
    trend: int = 73,
) -> None:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.set_index("date")
    value_cols = list(df.columns)
    trend_parts = []
    seasonal_parts = []
    resid_parts = []
    for col in value_cols:
        series = df[col]
        stl = STL(series, period=period, seasonal=seasonal, trend=trend, robust=True)
        result = stl.fit()
        trend_parts.append(result.trend.to_frame(name=col))
        seasonal_parts.append(result.seasonal.to_frame(name=col))
        resid_parts.append(result.resid.to_frame(name=col))
    trend_df = pd.concat(trend_parts, axis=1)
    seasonal_df = pd.concat(seasonal_parts, axis=1)
    resid_df = pd.concat(resid_parts, axis=1)
    trend_df.to_csv(trend_path, index_label="date")
    seasonal_df.to_csv(seasonal_path, index_label="date")
    resid_df.to_csv(resid_path, index_label="date")


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    dataset_dir = base / "dataset"
    input_csv = dataset_dir / "ETTh1_OT_augmented.csv"
    trend_csv = dataset_dir / "OT_trend.csv"
    seasonal_csv = dataset_dir / "OT_seasonal.csv"
    resid_csv = dataset_dir / "OT_resid.csv"
    decompose_etth1(input_csv, trend_csv, seasonal_csv, resid_csv)


if __name__ == "__main__":
    main()
