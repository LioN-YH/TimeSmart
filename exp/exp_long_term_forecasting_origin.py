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
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.best_val_loss = float("inf")

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

    # 判断当前是不是【多数据集分开验证】模式
    # 如果 vali_data 和 vali_loader 是列表（list）
    # → 说明是多个数据集分开验证（ _build_split_eval_datasets 返回的就是列表）
    # 如果是普通对象
    # → 单数据集验证
    def _is_split_eval(self, data_obj, loader_obj):
        return isinstance(data_obj, list) and isinstance(loader_obj, list)

    # 单个数据集的验证逻辑
    def _single_vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        # 切换评估模式
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

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
                total_loss.append(loss.item())

        total_loss = (
            float(np.average(total_loss)) if total_loss else float("inf")
        )  # 平均损失
        self.model.train()
        return total_loss

    def vali(self, vali_data, vali_loader, criterion):
        return self._single_vali(vali_data, vali_loader, criterion)

    def _multi_vali(self, eval_sets, eval_loaders, criterion, tag="val"):
        losses = {}
        weighted_loss = 0.0
        total_weight = 0.0
        # 遍历每个数据集
        for (name, ds, weight), (_, loader, _) in zip(eval_sets, eval_loaders):
            # 单独验证计算损失
            loss = self._single_vali(ds, loader, criterion)
            losses[name] = loss
            weighted_loss += weight * loss
            total_weight += weight
        # 计算加权平均损失
        if total_weight > 0:
            weighted_loss /= total_weight
        # 打印：每个数据集的单独损失 + 加权总分
        print(
            f"[{tag}] per-dataset losses: {losses}; weighted_{tag}_loss={weighted_loss:.7f}"
        )
        return weighted_loss, losses

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

        split_eval = self._is_split_eval(vali_data, vali_loader)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            # hasattr 检查是不是混合数据集 → 如果是，每轮 epoch 都刷新打乱一次
            if hasattr(train_data, "refresh_plan"):
                train_data.refresh_plan()

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

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

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

                if (i + 1) % 20 == 0:
                    print(
                        "	iters: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "	speed: {:.4f}s/iter; left time: {:.4f}s".format(
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

            if split_eval:
                vali_loss, vali_loss_detail = self._multi_vali(
                    vali_data, vali_loader, criterion, tag="val"
                )
                test_loss, test_loss_detail = self._multi_vali(
                    test_data, test_loader, criterion, tag="test"
                )
                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Weighted Vali Loss: {3:.7f} Weighted Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss
                    )
                )
            else:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)
                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss
                    )
                )

            if vali_loss < self.best_val_loss:
                self.best_val_loss = vali_loss

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        print("best model save path: {}".format(best_model_path))
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device)
        )

        return self.model

    def _test_single(self, setting, test_data, test_loader, file_suffix=""):
        preds = []
        trues = []

        vis_folder = os.path.join(
            self.args.results_folder, "test_results", setting + file_suffix
        )
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

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

                if getattr(test_data, "scale", False) and self.args.inverse:
                    shape = outputs.shape
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
                    input_arr = batch_x.detach().cpu().numpy()
                    if getattr(test_data, "scale", False) and self.args.inverse:
                        shape = input_arr.shape
                        input_arr = test_data.inverse_transform(
                            input_arr.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                    gt = np.concatenate((input_arr[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_arr[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(vis_folder, str(i) + ".pdf"))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        result_folder = os.path.join(
            self.args.results_folder, "results", setting + file_suffix
        )
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 20 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw_score = np.array(dtw_list).mean()
        else:
            dtw_score = "not calculated"

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse: {}, mae: {}, dtw: {}".format(mse, mae, dtw_score))

        results_path = self.args.results_path
        file_path = os.path.join(self.args.results_folder, results_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a") as f:
            f.write(setting + file_suffix + "  \n")
            f.write("mse: {}, mae: {}, dtw: {}".format(mse, mae, dtw_score))
            f.write("\n\n")

        np.save(
            os.path.join(result_folder, "metrics.npy"),
            np.array([mae, mse, rmse, mape, mspe]),
        )
        np.save(os.path.join(result_folder, "pred.npy"), preds)
        np.save(os.path.join(result_folder, "true.npy"), trues)

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "mspe": mspe,
            "dtw": dtw_score,
        }

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        if test:
            print("loading model")
            path = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(
                torch.load(
                    os.path.join(path, "checkpoint.pth"), map_location=self.device
                )
            )

        if self._is_split_eval(test_data, test_loader):
            summary = {}
            weighted_mse = 0.0
            weighted_mae = 0.0
            total_weight = 0.0
            for (name, ds, weight), (_, loader, _) in zip(test_data, test_loader):
                suffix = f"__{name}"
                metrics = self._test_single(setting, ds, loader, file_suffix=suffix)
                summary[name] = metrics
                weighted_mse += weight * metrics["mse"]
                weighted_mae += weight * metrics["mae"]
                total_weight += weight
            if total_weight > 0:
                weighted_mse /= total_weight
                weighted_mae /= total_weight
            print(
                f"weighted test summary: mse={weighted_mse}, mae={weighted_mae}, details={summary}"
            )
            return summary

        return self._test_single(setting, test_data, test_loader)
