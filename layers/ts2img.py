# 时序图像化代码
# 该代码将时间序列数据转换为图像格式，以便进行进一步的处理和分析

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Literal
from pyts.image import GramianAngularField, RecurrencePlot
import pywt
from scipy.signal import ShortTimeFFT

# 支持的时序图像化方法
ts2img_methods = ["Original", "SEG", "Plot", "GAF", "RP", "STFT", "WT"]


# 获取多种时序图像处理的张量结果
def get_ts2img_tensor(x_enc, device, args):
    H = args.image_size
    W = args.image_size
    ts2img_tensor_list = []
    for ts2img_method in ts2img_methods:
        ts2img_tensor = TimeSeriesToImage(
            x_enc=x_enc,
            H=H,
            W=W,
            ts2img_method=ts2img_method,
            device=device,
            args=args,
        )
        ts2img_tensor_list.append(ts2img_tensor)
    return ts2img_tensor_list


# TimeSeriesToImage函数：主控函数，根据传入参数，选择不同的图像化方式
def TimeSeriesToImage(x_enc, H, W, ts2img_method, device, args):

    if ts2img_method == "Original":
        original_transform = Original_Transform(
            hidden_dim=48,
            output_channels=3,
            periodicity=args.periodicity,
            H=H,
            W=W,
            device=device,
            compress_vars=args.compress_vars,
        )
        return original_transform(x_enc)

    elif ts2img_method == "SEG":
        return TimesNet_Transform(
            x_enc=x_enc,
            H=H,
            W=W,
            device=device,
            periodicity=args.periodicity,
            compress_vars=args.compress_vars,
            use_color=args.use_color,
        )

    elif ts2img_method == "Plot":
        return Plot_Transform(
            x_enc=x_enc,
            H=H,
            W=W,
            device=device,
            compress_vars=args.compress_vars,
            keep_labels=args.keep_labels,
            use_color=args.use_color,
        )

    elif ts2img_method == "GAF":
        return GramianAngularField_Transform(
            x_enc=x_enc,
            H=H,
            W=W,
            device=device,
            method=args.method,
            compress_vars=args.compress_vars,
            keep_labels=args.keep_labels,
            use_color=args.use_color,
        )

    elif ts2img_method == "RP":
        return RecurrencePlot_Transform(
            x_enc=x_enc,
            H=H,
            W=W,
            device=device,
            threshold=args.threshold,
            percentage=args.percentage,
            dimension=args.dimension,
            time_delay=args.time_delay,
            compress_vars=args.compress_vars,
            keep_labels=args.keep_labels,
            use_color=args.use_color,
        )

    elif ts2img_method == "STFT":
        return STFT_Transform(
            x_enc=x_enc,
            H=H,
            W=W,
            device=device,
            window_size=args.window_size,
            hop=args.hop,
            T_x=args.T_x,
            window_type=args.window_type,
            beta=args.beta,
            compress_vars=args.compress_vars,
            keep_labels=args.keep_labels,
            use_color=args.use_color,
        )

    elif ts2img_method == "WT":

        # 解析scales参数（格式为"start,end"）
        scales_start, scales_end = map(int, args.scales.split(","))
        scales = np.arange(scales_start, scales_end + 1)

        return Wavelet_Transform(
            x_enc=x_enc,
            H=H,
            W=W,
            device=device,
            scales=scales,
            wavelet=args.wavelet,
            compress_vars=args.compress_vars,
            keep_labels=args.keep_labels,
            use_color=args.use_color,
        )

    else:
        # 未支持的转换类型
        raise ValueError(f"Unknown ts2img_method:: {args.ts2img_method}")


# 压缩变量维度，沿着变量维度求平均
def Compress_Variables(images, B, nvars):
    images = einops.rearrange(
        images, "(b n) c h w -> b n c h w", b=B, n=nvars
    )  # [B, nvars, C, H, W]
    images = images.mean(dim=1)  # [B, C, H, W]
    return images


# TimesNet方法：根据数据的周期性对时序数据进行切分重排，构成2D图像
def TimesNet_Transform(
    x_enc,
    H,
    W,
    device,
    periodicity=None,
    compress_vars: bool = True,
    use_color: bool = True,
):

    B, seq_len, nvars = x_enc.shape  # 获取输入形状

    # 若periodicity为None，则通过FFT分析获取主要周期
    if periodicity is None:
        # 从输入数据中提取第一个批次的第一个单变量序列用于FFT分析
        x_fft = x_enc[0, :, 0]

        n = x_fft.size(0)
        fft_vals = torch.fft.rfft(x_fft)  # 实数FFT
        amplitudes = torch.abs(fft_vals)  # 振幅（频率强度）
        amplitudes[0] = 0  # 忽略直流分量（0频率，无周期意义）

        # 计算正频率（与numpy.fft.rfftfreq等价）
        freqs = torch.fft.rfftfreq(n, device=x_fft.device)

        # 找到振幅最大的频率（主要周期对应的频率）
        if torch.sum(amplitudes) == 0:
            # 若所有频率振幅为0（常数序列，无任何波动，无法识别周期），默认周期设为1
            periodicity = 1
        else:
            top_freq_idx = torch.argmax(amplitudes)  # 最大振幅对应的频率索引
            if freqs[top_freq_idx] == 0:
                # 避免除以0（理论上amplitudes[0]已置0，此处为冗余判断）
                periodicity = 1
            else:
                # 周期 = 1 / 频率（取整数）
                periodicity = int(1 / freqs[top_freq_idx].item())

        # 防止周期过大或过小
        min_period = 1
        max_period = seq_len // 2  # 最大周期不超过序列长度的一半
        periodicity = np.clip(periodicity, min_period, max_period)
        print(f"FFT_for_Period: {periodicity}")

    # Adjust padding to make seq_len a multiple of periodicity
    # 调整填充以使 seq_len 成为 periodicity 的倍数
    pad_left = 0
    if seq_len % periodicity != 0:
        # 计算需要填充的左侧长度 pad_left
        # 使得 (seq_len + pad_left) 可以被 periodicity 整除
        pad_left = periodicity - seq_len % periodicity

    # Rearrange to [B, nvars, seq_len]
    x_enc = einops.rearrange(x_enc, "b s n -> b n s")

    # Pad the time series
    # 在序列左侧进行复制边缘值的填充
    x_pad = F.pad(x_enc, (pad_left, 0), mode="replicate")
    # Reshape to [B * nvars, 1, f, p]
    # 根据周期进行截断重排
    x_2d = einops.rearrange(x_pad, "b n (p f) -> (b n) 1 f p", f=periodicity)

    # Resize the time series data
    # 重塑形状
    x_resized_2d = F.interpolate(
        x_2d, size=(H, W), mode="bilinear", align_corners=False
    )

    # Convert to 3-channel image
    if use_color:
        # 绘制为热力图
        x_resized_2d = x_resized_2d.squeeze(1).cpu()  # [B * nvars, H, W]
        images = []
        for i in range(x_resized_2d.shape[0]):
            fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
            im = ax.imshow(x_resized_2d[i], cmap="viridis", aspect="auto")

            # 移除坐标轴和边框
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # 调整布局，去除边距
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)

            # 保存图像到文件（可选）
            # plt.savefig(f"SEG_Heat_{i}.png", format="png", bbox_inches="tight", pad_inches=0)

            # 使用内存缓冲区保存图像
            with BytesIO() as buf:
                plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                buf.seek(0)

                # 从缓冲区读取图像并转换为张量
                with Image.open(buf) as img:
                    img = img.convert("RGB")  # 转换为RGB通道
                    # 确保图像尺寸正确
                    if img.size != (W, H):
                        img = img.resize((W, H), Image.Resampling.LANCZOS)
                    # 转化为张量
                    transform_img = transforms.ToTensor()
                    img_tensor = transform_img(img)
                    images.append(img_tensor)

            # 关闭图形释放内存
            plt.close(fig)

        # 堆叠所有图像张量，形成 [B*nvars, 3, H, W] 的输出
        images = torch.stack(images, dim=0).to(device)
    else:
        # 保留单通道设置，同时将像素值映射到[0, 255]
        images = normalize_images(x_resized_2d).to(device)  # [B*nvars, 1, H, W]

    # 压缩变量维度（可选）
    if compress_vars:
        images = Compress_Variables(images, B, nvars)  # [B, C, H, W]

    return images


# Plot方法：绘制时间序列数据的折线图
def Plot_Transform(
    x_enc,
    H,
    W,
    device,
    compress_vars: bool = True,
    keep_labels: bool = False,
    use_color: bool = True,
):

    if x_enc.device.type != "cpu":
        x_enc = x_enc.cpu()

    B, seq_len, nvars = x_enc.shape

    if use_color:
        # 使用 matplotlib 默认颜色循环
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        default_colors = prop_cycle.by_key()["color"]
        colors = [default_colors[i % len(default_colors)] for i in range(nvars)]
        convert_mode = "RGB"
    else:
        # 如果不使用彩色，则所有变量都用黑色
        colors = ["black"] * nvars
        convert_mode = "L"

    # 创建时间点
    time_points = torch.arange(seq_len, device="cpu")

    images = []

    # 设置matplotlib不显示图形窗口，节省资源
    plt.ioff()

    # 绘制折线图
    for b in range(B):
        for n in range(nvars):
            fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
            ax.plot(
                time_points,
                x_enc[b, :, n],
                linestyle="-",
                linewidth=1,
                marker="",
                markersize=4,
                color=colors[n],
            )

            if not keep_labels:
                # 移除坐标轴和边框
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                # 设置坐标轴标签
                ax.set_xlabel("Timestep", size=12)
                ax.set_ylabel("Value", size=12)
                ax.tick_params(axis="both", which="major", labelsize=10)

            # 调整布局，去除边距
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)

            # 保存图像到文件（可选）
            # plt.savefig(f"Plot_{b}_{n}.png", format="png", bbox_inches="tight", pad_inches=0)

            # 使用内存缓冲区保存图像
            with BytesIO() as buf:
                plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                buf.seek(0)

                # 从缓冲区读取图像并转换为张量
                with Image.open(buf) as img:
                    img = img.convert(convert_mode)
                    # 确保图像尺寸正确
                    if img.size != (W, H):
                        img = img.resize((W, H), Image.Resampling.LANCZOS)
                    # 转化为张量 (C, H, W)
                    transform_img = transforms.ToTensor()
                    img_tensor = transform_img(img)
                    images.append(img_tensor)

            plt.close(fig)

    # 堆叠所有图像张量，形成 [B*nvars, C, H, W] 的输出
    images = torch.stack(images, dim=0).to(device)
    # 压缩变量维度（可选）
    if compress_vars:
        images = Compress_Variables(images, B, nvars)  # [B, C, H, W]
    return images


# GramianAngularField方法：将时间序列转换为GAF图像
def GramianAngularField_Transform(
    x_enc,
    H,
    W,
    device,
    method: Literal["summation", "difference"] = "summation",
    compress_vars: bool = True,
    keep_labels: bool = False,
    use_color: bool = True,
):
    # 将Tensor转换为NumPy数组
    x_np = x_enc.cpu().detach().numpy()
    B, seq_len, nvars = x_np.shape

    # 初始化GAF转换器
    gaf_transformer = GramianAngularField(method=method, image_size=1.0)

    color = "gray"
    convert_mode = "L"
    # 颜色设置
    if use_color:
        color = "rainbow"
        convert_mode = "RGB"

    images = []

    for b in range(B):
        for n in range(nvars):

            # 提取单变量序列并调整形状
            single_series = x_np[b, :, n]
            single_series_reshaped = single_series.reshape(1, -1)  # [1, seq_len]

            # 计算GAF矩阵
            gaf = gaf_transformer.fit_transform(single_series_reshaped)

            # 绘制GAF图
            fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
            ax.imshow(gaf[0], cmap=color, origin="lower")

            if not keep_labels:
                # 移除坐标轴和边框
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                # 设置坐标轴标签
                ax.xlabel("Timestep", size=12)
                ax.ylabel("Timestep", size=12)
                ax.tick_params(axis="both", which="major", labelsize=10)

            # 调整布局，去除边距
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)

            # 保存图像到文件（可选）
            # plt.savefig(f"GAF_{b}_{n}.png", format="png", bbox_inches="tight", pad_inches=0)

            # 使用内存缓冲区保存图像
            with BytesIO() as buf:
                plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                buf.seek(0)

                # 从缓冲区读取图像并转换为张量
                with Image.open(buf) as img:
                    img = img.convert(convert_mode)
                    # 确保图像尺寸正确
                    if img.size != (W, H):
                        img = img.resize((W, H), Image.Resampling.LANCZOS)
                    # 转化为张量
                    transform_img = transforms.ToTensor()
                    img_tensor = transform_img(img)
                    images.append(img_tensor)

            # 关闭图形释放内存
            plt.close(fig)

    # 堆叠所有图像张量，形成 [B*nvars, C, H, W] 的输出
    images = torch.stack(images, dim=0).to(device)

    # 压缩变量维度（可选）
    if compress_vars:
        images = Compress_Variables(images, B, nvars)  # [B, C, H, W]
    return images


# RecurrencePlot方法：将时间序列转换为递归图
def RecurrencePlot_Transform(
    x_enc,
    H,
    W,
    device,
    threshold="point",
    percentage=10,
    dimension=1,
    time_delay=1,
    compress_vars: bool = True,
    keep_labels: bool = False,
    use_color: bool = True,
):
    # 将Tensor转换为NumPy数组
    x_np = x_enc.cpu().detach().numpy()
    B, seq_len, nvars = x_np.shape

    # 初始化递归图转换器
    transformer = RecurrencePlot(
        threshold=threshold,
        percentage=percentage,
        dimension=dimension,
        time_delay=time_delay,
    )

    color = "gray"
    convert_mode = "L"
    # 颜色设置
    if use_color:
        color = "rainbow"
        convert_mode = "RGB"

    images = []

    # 遍历每个批次和变量
    for b in range(B):
        for n in range(nvars):

            # 提取单变量序列并调整形状为 (1, seq_len)
            single_series = x_np[b, :, n]
            single_series_reshaped = single_series.reshape(1, -1)

            # 计算递归图
            rp = transformer.fit_transform(single_series_reshaped)

            # 绘制递归图
            fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
            ax.imshow(rp[0], cmap=color, origin="lower")

            if not keep_labels:
                # 移除坐标轴刻度和边框
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                # 设置坐标轴标签
                ax.xlabel("Timestep", size=12)
                ax.ylabel("Timestep", size=12)
                ax.tick_params(axis="both", which="major", labelsize=10)

            # 调整布局，去除所有边距
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)

            # 保存图像到文件（可选）
            # plt.savefig(f"RP_{b}_{n}.png", format="png", bbox_inches="tight", pad_inches=0)

            # 使用内存缓冲区保存图像
            with BytesIO() as buf:
                plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                buf.seek(0)

                # 从缓冲区读取图像并转换为张量
                with Image.open(buf) as img:
                    img = img.convert(convert_mode)
                    # 确保图像尺寸正确
                    if img.size != (W, H):
                        img = img.resize((W, H), Image.Resampling.LANCZOS)
                    # 转化为张量
                    transform_img = transforms.ToTensor()
                    img_tensor = transform_img(img)
                    images.append(img_tensor)

            # 关闭图形释放内存
            plt.close(fig)

    # 堆叠所有图像张量，形成 [B*nvars, C, H, W] 的输出
    images = torch.stack(images, dim=0).to(device)

    # 压缩变量维度（可选）
    if compress_vars:
        images = Compress_Variables(images, B, nvars)  # [B, C, H, W]
    return images


# STFT方法：将时间序列转换为短时傅里叶变换时频图
def STFT_Transform(
    x_enc,
    H,
    W,
    device,
    window_size: int = 20,
    hop: int = 10,
    T_x: float = 3600,
    window_type: str = "hann",
    beta: int = 14,
    compress_vars: bool = True,
    keep_labels: bool = False,
    use_color: bool = True,
):

    # 将Tensor转换为NumPy数组
    x_np = x_enc.cpu().detach().numpy()
    B, seq_len, nvars = x_np.shape

    # 初始化STFT
    # 根据窗口类型生成窗口函数
    if window_type == "rectangular":
        w = np.ones(window_size)
    elif window_type == "hann":
        w = np.hanning(window_size)
    elif window_type == "hamming":
        w = np.hamming(window_size)
    elif window_type == "blackman":
        w = np.blackman(window_size)
    elif window_type == "kaiser":
        w = np.kaiser(window_size, beta)
    else:
        raise ValueError(f"Invalid window type: {window_type}")

    SFT = ShortTimeFFT(w, hop=hop, fs=1 / T_x)

    color = "gray"
    convert_mode = "L"
    # 颜色设置
    if use_color:
        color = "viridis"
        convert_mode = "RGB"

    images = []

    # 遍历每个批次和变量
    for b in range(B):
        for n in range(nvars):
            # 提取单变量序列
            single_series = x_np[b, :, n]

            # 计算STFT
            Sx = SFT.stft(single_series)
            # 取对数幅度谱（加小值避免log(0)），压缩动态范围，使弱信号更明显
            stft_data = np.log(np.abs(Sx) + 0.01)

            # 绘制时频图
            fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
            im = ax.imshow(
                stft_data,
                origin="lower",
                aspect="auto",
                extent=[0, len(single_series), 0.0, SFT.extent(len(single_series))[-1]],
                cmap=color,
            )

            # 处理坐标轴和标签
            if not keep_labels:
                # 移除坐标轴刻度和边框
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                # 设置坐标轴标签
                ax.set_xlabel("Timestep", size=12)
                ax.set_ylabel("Freq (Hz)", size=12)
                ax.tick_params(axis="both", which="major", labelsize=10)
                # 科学计数法显示频率
                ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

            # 调整布局，去除边距
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)

            # 保存图像到文件（可选）
            # plt.savefig(
            #     f"STFT_{b}_{n}.png", format="png", bbox_inches="tight", pad_inches=0
            # )

            # 使用内存缓冲区保存图像
            with BytesIO() as buf:
                plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                buf.seek(0)

                # 从缓冲区读取图像并转换为张量
                with Image.open(buf) as img:
                    img = img.convert(convert_mode)
                    # 确保图像尺寸正确
                    if img.size != (W, H):
                        img = img.resize((W, H), Image.Resampling.LANCZOS)
                    # 转换为张量
                    transform_img = transforms.ToTensor()
                    img_tensor = transform_img(img)
                    images.append(img_tensor)

            # 关闭图形释放内存
            plt.close(fig)

    # 堆叠所有图像张量，形成 [B*nvars, C, H, W] 的输出
    images = torch.stack(images, dim=0).to(device)
    # 压缩变量维度（可选）
    if compress_vars:
        images = Compress_Variables(images, B, nvars)  # [B, C, H, W]
    return images


# Wavelet方法：将时间序列转换为小波变换时频图
def Wavelet_Transform(
    x_enc,
    H,
    W,
    device,
    scales: np.ndarray = np.arange(1, 400),
    wavelet: str = "morl",
    compress_vars: bool = True,
    keep_labels: bool = False,
    use_color: bool = True,
):
    # 将Tensor转换为NumPy数组
    x_np = x_enc.cpu().detach().numpy()
    B, seq_len, nvars = x_np.shape

    color = "gray"
    convert_mode = "L"
    # 颜色设置
    if use_color:
        color = "viridis"
        convert_mode = "RGB"

    images = []

    # 遍历每个批次和变量
    for b in range(B):
        for n in range(nvars):

            # 提取单变量序列
            single_series = x_np[b, :, n]

            # 计算小波变换
            coefficients, frequencies = pywt.cwt(
                data=single_series, scales=scales, wavelet=wavelet
            )
            # 使用小波系数的绝对值作为可视化数据（表示信号能量强度）
            wavelet_data = np.abs(coefficients)

            # 绘制小波变换图
            fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
            im = ax.imshow(
                wavelet_data,
                origin="upper",  # 小波图通常原点在上（与STFT不同）
                aspect="auto",
                extent=[0, len(single_series), 0, len(scales)],
                cmap=color,
            )

            # 处理坐标轴和标签
            if not keep_labels:
                # 移除坐标轴刻度和边框
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                # 设置坐标轴标签
                ax.set_xlabel("Timestep", size=12)
                ax.set_ylabel("Scale", size=12)
                ax.tick_params(axis="both", which="major", labelsize=10)
                # 科学计数法显示y轴
                ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

            # 小波图通常需要翻转y轴（尺度从大到小）
            ax.invert_yaxis()

            # 调整布局，去除边距
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)

            # 保存图像到文件（可选）
            # plt.savefig(
            #     f"Wavelet_{b}_{n}.png", format="png", bbox_inches="tight", pad_inches=0
            # )

            # 将图像保存到内存缓冲区
            with BytesIO() as buf:
                plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
                buf.seek(0)

                # 转换为图像张量
                with Image.open(buf) as img:
                    img = img.convert(convert_mode)
                    # 确保图像尺寸正确
                    if img.size != (W, H):
                        img = img.resize((W, H), Image.Resampling.LANCZOS)
                    # 转换为张量
                    transform_img = transforms.ToTensor()
                    img_tensor = transform_img(img)
                    images.append(img_tensor)

            # 关闭图形释放内存
            plt.close(fig)

    # 堆叠所有图像张量，形成 [B*nvars, C, H, W] 的输出
    images = torch.stack(images, dim=0).to(device)
    # 压缩变量维度（可选）
    if compress_vars:
        images = Compress_Variables(images, B, nvars)  # [B, C, H, W]
    return images


# TimeVLM论文方法：原始序列+FFT+周期性编码，经过可学习的卷积网络转换为图像
# 此处进行小幅度修改，以完成维度适配
class Original_Transform(nn.Module):
    """Learnable module to convert time series data into image tensors"""

    def __init__(
        self,
        hidden_dim,
        output_channels,
        periodicity,
        H,
        W,
        device,
        compress_vars: bool = True,
    ):
        super(Original_Transform, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.periodicity = periodicity
        self.H = H
        self.W = W
        self.device = device
        self.compress_vars = compress_vars

        # 1D convolutional layer
        self.conv1d = nn.Conv1d(
            in_channels=4, out_channels=hidden_dim, kernel_size=3, padding=1
        )

        # 2D convolution layers
        self.conv2d_1 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim // 2,
            kernel_size=3,
            padding=1,
        )
        self.conv2d_2 = nn.Conv2d(
            in_channels=hidden_dim // 2,
            out_channels=output_channels,
            kernel_size=3,
            padding=1,
        )

        self.conv1d = self.conv1d.to(self.device)
        self.conv2d_1 = self.conv2d_1.to(self.device)
        self.conv2d_2 = self.conv2d_2.to(self.device)

    def forward(self, x_enc):
        """Convert input time series to image tensor [B, output_channels, H, W]"""
        B, L, D = x_enc.shape

        # Generate periodicity encoding (sin/cos)
        time_steps = (
            torch.arange(L, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(B, 1)
            .to(x_enc.device)
        )
        periodicity_encoding = torch.cat(
            [
                torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
                torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
            ],
            dim=-1,
        )
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(
            1, 1, D, 1
        )  # [B, L, D, 2]

        # FFT frequency encoding (magnitude)
        x_fft = torch.fft.rfft(x_enc, dim=1)
        x_fft_mag = torch.abs(x_fft)
        # 零填充操作
        # 如果变换后的频谱长度（L//2+1）小于目标长度L，则在右侧填充零
        if x_fft_mag.shape[1] < L:
            pad = torch.zeros(
                B, L - x_fft_mag.shape[1], D, device=x_enc.device, dtype=x_fft_mag.dtype
            )
            x_fft_mag = torch.cat([x_fft_mag, pad], dim=1)
        x_fft_mag = x_fft_mag.unsqueeze(-1)  # [B, L, D, 1]

        # Combine all features: raw + FFT + periodicity
        x_enc = x_enc.unsqueeze(-1)  # [B, L, D, 1]
        x_enc = torch.cat(
            [x_enc, x_fft_mag, periodicity_encoding], dim=-1
        )  # [B, L, D, 4]

        # Reshape for 1D convolution
        # 一维卷积
        x_enc = x_enc.permute(0, 2, 3, 1)  # [B, D, 4, L]
        x_enc = x_enc.reshape(B * D, 4, L)  # [B*D, 4, L]
        x_enc = self.conv1d(x_enc)  # [B*D, hidden_dim, L]

        # 增加一个新的维度
        x_enc = x_enc.unsqueeze(2)  # [B*D, hidden_dim, 1, L]

        # 2D Convolution processing
        # 二维卷积
        x_enc = F.tanh(self.conv2d_1(x_enc))
        x_enc = F.tanh(self.conv2d_2(x_enc))  # [B*D, output_channels, 1, L]

        # Resize to target image size
        # 重塑图像
        x_enc = F.interpolate(
            x_enc, size=(self.H, self.W), mode="bilinear", align_corners=False
        )  # [B*D, output_channels, H, W]

        # 压缩变量维度（可选）
        if self.compress_vars:
            x_enc = Compress_Variables(x_enc, B, D)  # [B, 3, H, W]

        # 图像归一化
        images = normalize_images(x_enc)
        return images


# 通过FFT识别周期
def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list

    # 周期和对应的频率振幅
    return period, abs(xf).mean(-1)[:, top_list]


# 判断一个形状为 [C, H, W] 的 Tensor 是否为灰度图像
def is_grayscale_tensor(tensor, tol=1e-6):

    if tensor.shape[0] == 1:
        return True
    elif tensor.shape[0] == 3:
        # 拆分三个通道
        r, g, b = tensor[0], tensor[1], tensor[2]

        # 检查 R 和 G 的差异，R 和 B 的差异
        diff_rg = torch.abs(r - g)
        diff_rb = torch.abs(r - b)

        # 如果所有差异都小于容忍度，则认为是灰度图
        return (diff_rg < tol).all() and (diff_rb < tol).all()
    else:
        raise ValueError("Warning: Unexpected number of channels")


# 图像归一化，将像素值映射到[0, 255]
def normalize_images(images):
    # Compute min and max per image across all channels and spatial dimensions
    # 1. 计算每张图像的最小值和最大值（跨所有通道和像素）
    min_vals = (
        images.reshape(images.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    )
    max_vals = (
        images.reshape(images.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    )

    # Avoid division by zero by adding a small epsilon
    # 2. 计算缩放因子，同时防止除零错误
    epsilon = 1e-5
    scale = (max_vals - min_vals).clamp(min=epsilon)

    # Normalize to [0, 1]
    # 3. 归一化图像到 [0, 1] 范围
    images = (images - min_vals) / scale

    # Scale to [0, 255] and clamp to ensure valid range
    # 转换到 [0, 255] 范围并转为整数
    images = (images * 255).clamp(0, 255).to(torch.uint8)

    return images
