import argparse
import os
import sys
import torch
import h5py
import numpy as np


current_script_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_script_path)
grandparent_dir = os.path.dirname(parent_dir)
project_root = os.path.dirname(grandparent_dir)
sys.path.append(project_root)

from torch.utils.data import DataLoader
from data_provider.data_loader_save import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_M4,
)

from layers.ts2img import TimeSeriesToImage

# 时序图像化方法列表
time2img_list = ["SEG", "Plot", "GAF", "RP", "STFT", "WT"]

# 数据划分列表
divide_list = ["train", "val", "test"]

# 数据集字典
data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
    "m4": Dataset_M4,
}


# 参数解析函数
def parse_args():
    parser = argparse.ArgumentParser()

    # 运行环境相关参数
    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )

    # 为了存储对应的图像化结果，batchsize设置为1，每个样本都进行预计算
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size of train input data"
    )

    # 数据加载相关参数
    parser.add_argument(
        "--data", type=str, required=True, default="ETTm1", help="dataset type"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, default="ETTm1", help="dataset name"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument(
        "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
    )
    parser.add_argument("--norm_const", type=float, default=0.4)
    parser.add_argument(
        "--augmentation_ratio", type=int, default=0, help="How many times to augment"
    )

    # 预测任务相关参数
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )

    # 时序图像化处理相关参数
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="image size for time series to image",
    )

    # 通用
    parser.add_argument(
        "--keep_labels",
        type=bool,
        default=False,
        help="Retain labels when plotting the images",
    )
    # TimesNet_Transform
    parser.add_argument(
        "--periodicity",
        type=int,
        default=96,
        help="The periodicity of the dataset, used for TimesNet_Transform",
    )
    # GramianAngularField_Transform
    parser.add_argument(
        "--method",
        type=str,
        default="summation",
        help="Type of GramianAngularField_Transform",
    )
    # RecurrencePlot_Transform
    parser.add_argument(
        "--threshold",
        type=str,
        default="point",
        help="Threshold for the minimum distance, used for RecurrencePlot_Transform",
    )
    parser.add_argument(
        "--dimension",
        type=float,
        default=1,
        help="Dimension of the trajectory for RecurrencePlot_Transform. If float, it represents a percentage of the size of each time series and must be between 0 and 1",
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=10,
        help="Percentage of black points if threshold='point' or percentage of maximum distance for threshold if threshold='distance', used for RecurrencePlot_Transform",
    )
    parser.add_argument(
        "--time_delay",
        type=float,
        default=1,
        help="Time gap between two back-to-back points of the trajectory used for RecurrencePlot_Transform. If float, it represents a percentage of the size of each time series and must be between 0 and 1",
    )
    # STFT_Transform
    parser.add_argument(
        "--window_size", type=int, default=20, help="Window size for STFT_Transform"
    )
    parser.add_argument(
        "--hop",
        type=int,
        default=10,
        help="Hop size between consecutive windows in STFT_Transform",
    )
    parser.add_argument(
        "--T_x",
        type=float,
        default=3600,
        help="Sampling interval of the dataset, used for STFT_Transform",
    )
    parser.add_argument(
        "--window_type",
        type=str,
        default="hann",
        help="Window type for STFT_Transform, options: rectangular, hann, hamming, blackman, kaiser",
    )
    parser.add_argument(
        "--beta",
        type=int,
        default=14,
        help="Beta parameter for kaiser window in STFT_Transform",
    )
    # Wavelet_Transform
    parser.add_argument(
        "--wavelet", type=str, default="morl", help="Wavelet type for Wavelet_Transform"
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="1,400",
        help='Scales for Wavelet_Transform, specified as "start,end" (default: 1,400)',
    )

    # 预计算文件保存路径
    parser.add_argument(
        "--save_folder",
        type=str,
        default="./ImageTensors/",
        help="folder to save the results",
    )
    args = parser.parse_args()
    return args


# 根据不同设置，返回data_set和data_loader
def data_provider(args, flag):
    # 根据传入的args.data选择对应的数据集类
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1

    # 测试时不打乱数据，训练/验证时打乱
    shuffle_flag = False if (flag == "test" or flag == "TEST") else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.data == "m4":
        drop_last = False
    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
    )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )

    # print(f"[DEBUG] DataLoader shuffle={shuffle_flag}, flag={flag}")
    return data_set, data_loader


# 数据归一化处理
def normalize_input(x, norm_const):
    # 计算均值
    means = x.mean(1, keepdim=True).detach()
    x = x - means
    # 计算标准差
    # unbiased=False：使用无偏估计，计算样本方差
    # 1e-5：防止除零错误-接近零的方差会导致标准差接近零，在反向传播时可能引发梯度爆炸
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    # 使用配置中的norm_const进一步缩放标准差
    stdev /= norm_const
    x = x / stdev
    return x


# 预计算图像张量，保存在本地
def precompute_tensors(args):

    for divide in divide_list:
        # 图片张量保存路径，类似于./ImageTensors/ETTh1/train/
        save_path = (
            args.save_folder + f"{args.dataset}/predLen_{args.pred_len}/{divide}"
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 获取数据集和数据加载器
        data_set, data_loader = data_provider(args, divide)

        # 遍历数据加载器，逐批处理数据
        # 已知batchsize=1，因此每次处理一个样本
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_index) in enumerate(
            data_loader
        ):
            batch_x = batch_x.float()
            index = batch_index[0].item()
            x = normalize_input(batch_x, args.norm_const)
            file_path = f"{save_path}/{index}.h5"
            with h5py.File(file_path, "w") as hf:
                # 获取张量，保存为h5文件
                for time2img in time2img_list:
                    img_tensor = TimeSeriesToImage(
                        x_enc=x,
                        H=args.image_size,
                        W=args.image_size,
                        time2img_type=time2img,
                        args=args,
                    )
                    hf.create_dataset(time2img, data=img_tensor.cpu().numpy())
                    # print(f"Saved {time2img} image tensor to {file_path}")
        print(
            f"Precomputed tensors for {args.dataset} {divide} set saved in {save_path}"
        )


if __name__ == "__main__":
    args = parse_args()
    precompute_tensors(args)
