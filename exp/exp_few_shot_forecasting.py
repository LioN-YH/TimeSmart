from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw

warnings.filterwarnings("ignore")


class Exp_Few_Shot_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Few_Shot_Forecast, self).__init__(args)

        # 初始化用于收集 gate 输出的容器
        self.gate_weights_collector = []
        # hook 句柄，便于后续移除
        self.hook_handle = None

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    # 注册hook函数
    def _register_hook(self):
        module = self.model.gate

        def hook_fn(module, input, output):
            # output shape: (batch_size, n_vars, 2)
            self.gate_weights_collector.append(output.detach().cpu().numpy())

        self.hook_handle = module.register_forward_hook(hook_fn)
        print(f">>> [Hook] Registered hook")

    # 移除hook函数
    def _remove_hook(self):
        """移除已注册的 hook，防止影响其他运行或内存泄漏"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            print(f">>> [Hook] Removed hook")

    # 统计多模态分支分配的平均权重
    def _calculate_multimodal_weight(self):
        multimodal_weight = 0
        # 合并所有 batch 的 gate weights
        if self.gate_weights_collector:
            gate_weights = np.concatenate(self.gate_weights_collector, axis=0)
            # 计算均值
            multimodal_weight = gate_weights[:, :, 1].mean()
        return multimodal_weight

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            # 训练开始前
            # 清空上一轮的 gate 输出记录
            self.gate_weights_collector = []
            # 注册hook
            self._register_hook()

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                            self.device
                        )
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # 训练结束后，移除hook
            self._remove_hook()
            # 计算多模态分支的平均权重
            multimodal_weight = self._calculate_multimodal_weight()

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} | Multimodal Weight: {5:.7f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    test_loss,
                    multimodal_weight,
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        print("best model save path: {}".format(best_model_path))
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        # 这里检查 test_data 是否为 torch.utils.data.Subset 类型
        # Subset 是 PyTorch 提供的一个包装类（wrapper），用于从一个完整数据集中抽取子集
        # 通常不直接包含原始数据的所有属性，而是通过 .dataset 指向原始数据集

        # 如果 test_data 是一个 Subset 实例，则不能直接从 test_data 获取 .scale 属性（因为 Subset 本身可能没有这个属性）
        # 需要访问其内部的原始数据集test_data.dataset，从而安全地获取数据的缩放（scaling）参数
        if isinstance(test_data, torch.utils.data.Subset):
            data_scaling = test_data.dataset.scale
        else:
            data_scaling = test_data.scale

        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(
                    os.path.join("./checkpoints/" + setting, "checkpoint.pth"),
                    map_location=self.device,
                )
            )

        # 清空之前的 gate 输出记录
        self.gate_weights_collector = []
        # 注册hook
        self._register_hook()

        preds = []
        trues = []
        folder_path = "./Result/test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # 与前边类似的，如果 test_data 是 Subset，则需要通过 test_data.dataset 访问原始数据集的方法
                if data_scaling and self.args.inverse:
                    shape = outputs.shape
                    if isinstance(test_data, torch.utils.data.Subset):
                        outputs = test_data.dataset.inverse_transform(
                            outputs.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                        batch_y = test_data.dataset.inverse_transform(
                            batch_y.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                    else:
                        outputs = test_data.inverse_transform(
                            outputs.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                        batch_y = test_data.inverse_transform(
                            batch_y.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if data_scaling and self.args.inverse:
                        shape = input.shape
                        if isinstance(test_data, torch.utils.data.Subset):
                            input = test_data.dataset.inverse_transform(
                                input.reshape(shape[0] * shape[1], -1)
                            ).reshape(shape)
                        else:
                            input = test_data.inverse_transform(
                                input.reshape(shape[0] * shape[1], -1)
                            ).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        # 训练结束后，移除hook
        self._remove_hook()
        # 计算多模态分支的平均权重
        multimodal_weight = self._calculate_multimodal_weight()

        # result save
        folder_path = "./Result/results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = "not calculated"

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(
            "mse: {}, mae: {}, dtw: {}, multimodal_weight: {}".format(
                mse, mae, dtw, multimodal_weight
            )
        )

        results_path = self.args.results_path
        file_path = os.path.join("Result", results_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        f = open(file_path, "a")
        f.write(setting + "  \n")
        f.write(
            "mse: {}, mae: {}, dtw: {}, multimodal_weight: {}".format(
                mse, mae, dtw, multimodal_weight
            )
        )
        f.write("\n")
        f.write("\n")
        f.close()

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)

        return
