import os
import torch
from models import (
    Autoformer,
    Transformer,
    TimesNet,
    Nonstationary_Transformer,
    DLinear,
    FEDformer,
    Informer,
    LightTS,
    Reformer,
    ETSformer,
    Pyraformer,
    PatchTST,
    MICN,
    Crossformer,
    FiLM,
    iTransformer,
    Koopa,
    TiDE,
    FreTS,
    TimeMixer,
    TSMixer,
    SegRNN,
    MambaSimple,
    TemporalFusionTransformer,
    SCINet,
    PAttn,
    TimeXer,
    TimeLLM,
    VisionTS,
)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "TimeLLM": TimeLLM,
            "VisionTS": VisionTS,
            "TimesNet": TimesNet,
            "Autoformer": Autoformer,
            "Transformer": Transformer,
            "Stationary": Nonstationary_Transformer,
            "DLinear": DLinear,
            "FEDformer": FEDformer,
            "Informer": Informer,
            "LightTS": LightTS,
            "Reformer": Reformer,
            "ETSformer": ETSformer,
            "PatchTST": PatchTST,
            "Pyraformer": Pyraformer,
            "MICN": MICN,
            "Crossformer": Crossformer,
            "FiLM": FiLM,
            "iTransformer": iTransformer,
            "Koopa": Koopa,
            "TiDE": TiDE,
            "FreTS": FreTS,
            "MambaSimple": MambaSimple,
            "TimeMixer": TimeMixer,
            "TSMixer": TSMixer,
            "SegRNN": SegRNN,
            "TemporalFusionTransformer": TemporalFusionTransformer,
            "SCINet": SCINet,
            "PAttn": PAttn,
            "TimeXer": TimeXer,
        }
        # 动态导入 TimeVLM 模型
        if args.model == "TimeVLM":
            from src.TimeVLM import model as TimeVLM

            self.model_dict["TimeVLM"] = TimeVLM

        # CHANGE:更改的模型
        elif args.model == "TimeSmart":
            from src.TimeSmart import model as TimeSmart

            self.model_dict["TimeSmart"] = TimeSmart

        elif args.model == "TimeVLM-wo-Mul":
            from src.TimeVLM import model_woMul as TimeVLM_woMul

            self.model_dict["TimeVLM-wo-Mul"] = TimeVLM_woMul

        elif args.model == "TimeVLM-Bi-Retrieval":
            from src.TimeSmart import model_Bi_Retrieval as TimeVLM_Bi_Retrieval

            self.model_dict["TimeVLM-Bi-Retrieval"] = TimeVLM_Bi_Retrieval

        elif args.model == "TimeSmart_moe_single":
            from src.TimeSmart import model_moe as TimeSmart_moe_single

            self.model_dict["TimeSmart_moe_single"] = TimeSmart_moe_single

        elif args.model == "TimeSmart_moe_v":
            from src.TimeSmart import model_moe_v as TimeSmart_moe_v

            self.model_dict["TimeSmart_moe_v"] = TimeSmart_moe_v

        elif args.model == "TimeSmart_test":
            from src.TimeSmart import model_test as TimeSmart_test

            self.model_dict["TimeSmart_test"] = TimeSmart_test

        elif args.model == "TimeVLM_v_l":
            from src.TimeVLM import model_v_l as TimeVLM_v_l

            self.model_dict["TimeVLM_v_l"] = TimeVLM_v_l

        elif args.model == "VLMtest":
            from src.TimeSmart import test as VLMtest

            self.model_dict["VLMtest"] = VLMtest

        elif args.model == "TimeSmart_top3":
            from src.TimeSmart import moe_top3 as TimeSmart_top3

            self.model_dict["TimeSmart_top3"] = TimeSmart_top3

        # 获取设备（GPU 或 CPU）
        self.device = self._acquire_device()
        # 构建模型并移动到指定设备
        self.model = self._build_model().to(self.device)

        if args.is_training:
            self._log_model_parameters()

    def _log_model_parameters(self):
        """
        打印模型参数。
        """

        # 使用 p.numel() 计算每个参数张量的元素个数
        # 统计模型可训练参数数量
        def count_learnable_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 统计模型总参数数量
        def count_total_parameters(model):
            return sum(p.numel() for p in model.parameters())

        learable_params = count_learnable_parameters(self.model)
        total_params = count_total_parameters(self.model)
        print(f"Learnable model parameters: {learable_params:,}")
        print(f"Total model parameters: {total_params:,}")

    # 构建模型
    def _build_model(self):
        raise NotImplementedError
        return None

    # 获取设备（GPU 或 CPU）
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
