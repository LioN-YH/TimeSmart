import pandas as pd
import plotly.express as px

# INTRO：可视化csv文件
# 支持：
# 鼠标滚轮缩放
# 拖拽平移
# 悬停查看各变量在某时刻的值
# 底部时间滑块快速定位时间段
# 点击图例隐藏/显示某个变量

csv_path = "/home/user10/TimeSmart/dataset/ETTh1.csv"

df = pd.read_csv(csv_path)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    time_col = "date"
elif "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    time_col = "Date"
else:
    non_numeric_cols = df.select_dtypes(exclude="number").columns
    time_col = non_numeric_cols[0] if len(non_numeric_cols) else df.columns[0]
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception:
        pass

numeric_cols = [
    c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])
]

df_long = df.melt(
    id_vars=[time_col], value_vars=numeric_cols, var_name="变量", value_name="值"
)

fig = px.line(df_long, x=time_col, y="值", color="变量", title="ETTh1 多变量时序")
fig.update_layout(hovermode="x unified", xaxis=dict(rangeslider=dict(visible=True)))
fig.show()
fig.write_html("etth1_plot.html", include_plotlyjs="cdn")
