from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_M4,
    PSMSegLoader,
    MSLSegLoader,
    SMAPSegLoader,
    SMDSegLoader,
    SWATSegLoader,
    UEAloader,
)
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import torch

# 数据集字典：字符串-数据集类
data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
    "m4": Dataset_M4,
    "PSM": PSMSegLoader,
    "MSL": MSLSegLoader,
    "SMAP": SMAPSegLoader,
    "SMD": SMDSegLoader,
    "SWAT": SWATSegLoader,
    "UEA": UEAloader,
}


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

    # 异常检测任务数据构造
    if args.task_name == "anomaly_detection":
        drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader
    # 分类任务数据构造
    elif args.task_name == "classification":
        drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),  # 处理变长序列
        )
        return data_set, data_loader
    # 预测任务数据构造
    else:
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
        # 少样本学习，并且为训练阶段
        if args.percent < 1.0 and flag == "train":
            num_samples = int(len(data_set) * args.percent)
            # 随机打乱并选取子集索引，实现了对训练集的随机采样
            # torch.randperm(n)：生成从 0 到 n-1 的随机排列的一个张量
            indices = torch.randperm(len(data_set))[:num_samples]
            # 构造子集，Subset 并不会复制数据，只是引用了原始数据中的某些索引
            data_set = torch.utils.data.Subset(data_set, indices)
            print(
                f"Few-shot sampling: {args.percent*100}% of data, {len(data_set)} samples"
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
