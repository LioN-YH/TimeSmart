from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    MixedForecastDataset,
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


# _build_forecast_dataset：构建单预测数据集 （工厂函数）
def _build_forecast_dataset(
    args,
    Data,
    flag,
    timeenc,
    freq,
    data_path,
    root_path=None,
    features=None,
    target=None,
    seasonal_patterns=None,
):
    return Data(
        args=args,
        root_path=root_path if root_path is not None else args.root_path,
        data_path=data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=features if features is not None else args.features,
        target=target if target is not None else args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=(
            seasonal_patterns
            if seasonal_patterns is not None
            else args.seasonal_patterns
        ),
    )


# _apply_few_shot_if_needed：小样本采样
def _apply_few_shot_if_needed(args, data_set, flag):
    if args.percent < 1.0 and flag == "train":
        num_samples = int(len(data_set) * args.percent)
        indices = torch.randperm(len(data_set))[:num_samples]
        data_set = torch.utils.data.Subset(data_set, indices)
        print(
            f"Few-shot sampling: {args.percent*100}% of data, {len(data_set)} samples"
        )
    return data_set


# _get_mix_list：解析混合数据集参数
def _get_mix_list(args, attr_name, fallback_value=None):
    values = getattr(args, attr_name, None)
    # 空值转为空列表
    if values is None:
        values = []
    # 字符串转列表（比如 "h,d,m" → ["h","d","m"]）
    if isinstance(values, str):
        values = [x.strip() for x in values.split(",") if x.strip()]
    #  空列表 → 用默认值填充，长度=混合数据集数量
    if len(values) == 0:
        values = [fallback_value] * len(args.mix_data_paths)
    if len(values) != len(args.mix_data_paths):
        raise ValueError(f"{attr_name} and mix_data_paths must have the same length.")
    return values


# _resolve_mix_specs：解析混合训练完整配置
def _resolve_mix_specs(args):
    mix_data_types = _get_mix_list(args, "mix_data_types", args.data)
    mix_root_paths = _get_mix_list(args, "mix_root_paths", args.root_path)
    mix_freqs = _get_mix_list(args, "mix_freqs", args.freq)
    mix_targets = _get_mix_list(args, "mix_targets", args.target)
    mix_features = _get_mix_list(args, "mix_features", args.features)

    specs = []  # 存储所有数据集的完整配置
    for i, data_path in enumerate(args.mix_data_paths):
        data_name = mix_data_types[i]
        if data_name not in data_dict:
            raise ValueError(
                f"Unknown mix_data_type '{data_name}' at index {i}. "
                f"Available keys: {sorted(data_dict.keys())}"
            )
        specs.append(
            {
                "name": args.mix_data_names[i],
                "data_path": data_path,
                "weight": args.mix_weights[i],
                "data_type": data_name,
                "Data": data_dict[data_name],
                "root_path": mix_root_paths[i],
                "freq": mix_freqs[i],
                "target": mix_targets[i],
                "features": mix_features[i],
            }
        )
    return specs


# 构建混合训练集：_build_mixed_train_dataset
def _build_mixed_train_dataset(args, timeenc):
    datasets = []
    mix_specs = _resolve_mix_specs(args)
    for spec in mix_specs:
        ds = _build_forecast_dataset(
            args,
            spec["Data"],
            "train",
            timeenc,
            spec["freq"],
            spec["data_path"],
            root_path=spec["root_path"],
            features=spec["features"],
            target=spec["target"],
        )
        datasets.append(ds)
        print(
            f"train-{spec['name']}: type={spec['data_type']}, path={spec['data_path']}, "
            f"root={spec['root_path']}, freq={spec['freq']}, samples={len(ds)}"
        )
    epoch_len = args.mix_epoch_len if args.mix_epoch_len > 0 else None
    mixed_dataset = MixedForecastDataset(
        datasets=datasets,
        weights=args.mix_weights,
        dataset_names=args.mix_data_names,
        epoch_len=epoch_len,
        seed=args.seed,
    )
    return mixed_dataset


# 构建验证 / 测试集：_build_split_eval_datasets
def _build_split_eval_datasets(args, flag, timeenc, batch_size, num_workers, drop_last):
    eval_sets = []
    eval_loaders = []
    mix_specs = _resolve_mix_specs(args)
    for spec in mix_specs:
        ds = _build_forecast_dataset(
            args,
            spec["Data"],
            flag,
            timeenc,
            spec["freq"],
            spec["data_path"],
            root_path=spec["root_path"],
            features=spec["features"],
            target=spec["target"],
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        eval_sets.append((spec["name"], ds, spec["weight"]))
        eval_loaders.append((spec["name"], loader, spec["weight"]))
        print(
            f"{flag}-{spec['name']}: type={spec['data_type']}, path={spec['data_path']}, "
            f"root={spec['root_path']}, freq={spec['freq']}, samples={len(ds)}"
        )
    return eval_sets, eval_loaders


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
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
        )
        return data_set, data_loader
    # 预测任务数据构造
    else:
        if args.data == "m4":
            drop_last = False

        if getattr(args, "use_mix_dataset", False):
            if flag == "train":
                data_set = _build_mixed_train_dataset(args, timeenc)
                print(flag, len(data_set))
                data_loader = DataLoader(
                    data_set,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    drop_last=drop_last,
                )
                return data_set, data_loader
            if flag in ["val", "test"]:
                return _build_split_eval_datasets(
                    args, flag, timeenc, batch_size, args.num_workers, drop_last
                )

        data_set = _build_forecast_dataset(
            args, Data, flag, timeenc, freq, args.data_path
        )
        data_set = _apply_few_shot_if_needed(args, data_set, flag)
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader
