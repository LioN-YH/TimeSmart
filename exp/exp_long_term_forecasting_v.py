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
import pandas as pd
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

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        # if hasattr(model_ref, "meta_mean"):
        #     print(f"vali meta_mean: {model_ref.meta_mean}")
        # if hasattr(model_ref, "meta_std"):
        #     print(f"vali meta_std: {model_ref.meta_std}")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
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
            self.model.train()
            epoch_time = time.time()
            model_ref = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            model_ref.meta_records = []  # 元特征记录
            model_ref.ts2img_records = []  # ts2img权重记录
            router_grad_row_norm_records = []  # 路由器梯度范数记录
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
                if (i + 1) % 10 == 0:
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
                # 梯度记录
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(model_optim)
                    if hasattr(model_ref, "router") and hasattr(model_ref.router, "fc"):
                        g = model_ref.router.fc.weight.grad  # 路由器全连接层的梯度
                        if g is not None:
                            router_grad_row_norm_records.append(
                                g.detach().norm(dim=1).float().cpu().numpy()
                            )
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if hasattr(model_ref, "router") and hasattr(model_ref.router, "fc"):
                        g = model_ref.router.fc.weight.grad
                        if g is not None:
                            router_grad_row_norm_records.append(
                                g.detach().norm(dim=1).float().cpu().numpy()
                            )
                    model_optim.step()
            folder_path = (
                self.args.results_folder
                + "train_results/"
                + setting
                + "/ts2img_weights/"
            )
            os.makedirs(folder_path, exist_ok=True)

            # ts2img权重记录
            w_list = [t.numpy() for t in model_ref.ts2img_records]
            w_arr = np.concatenate(w_list, axis=0) if len(w_list) > 0 else np.array([])
            np.save(os.path.join(folder_path, f"epoch_{epoch + 1}.npy"), w_arr)

            diag_folder = (
                self.args.results_folder + "train_results/" + setting + "/router_diag/"
            )
            os.makedirs(diag_folder, exist_ok=True)

            # epoch_X.npz ：选择行为诊断
            # 包含：entropy_mean(平均熵：衡量 Router 输出概率分布的不确定性),
            #      top1_mean(Top-1 概率均值：Router 输出的最高概率值的平均数),
            #      method_hist(选择直方图：统计了每个可视化方法被选为 Top-1 的频率)
            if w_arr.size > 0 and len(w_arr.shape) == 3:
                p = w_arr.reshape(-1, w_arr.shape[-1]).astype(np.float64)
                entropy = (-(p * np.log(p + 1e-12)).sum(axis=-1)).mean()
                top1 = p.max(axis=-1).mean()
                idx = p.argmax(axis=-1)
                counts = np.bincount(idx, minlength=p.shape[-1]).astype(np.float64)
                counts = counts / max(1, idx.size)
                np.savez(
                    os.path.join(diag_folder, f"epoch_{epoch + 1}.npz"),
                    entropy_mean=np.array(entropy, dtype=np.float32),
                    top1_mean=np.array(top1, dtype=np.float32),
                    method_hist=counts.astype(np.float32),
                )

            # epoch_X_grad.npz ：梯度健康诊断
            # 包含：grad_row_norm(梯度范数矩阵：记录了该 Epoch 内每个 Step、每个 Expert 分支对应的梯度范数【num_sample, d_ts2img】),
            #      grad_row_norm_mean(平均梯度范数：每个 Expert 接收到的平均梯度大小),
            #      grad_row_norm_std(梯度范数标准差：梯度的波动情况)
            if len(router_grad_row_norm_records) > 0:
                grad_arr = np.stack(router_grad_row_norm_records, axis=0).astype(
                    np.float32
                )
                np.savez(
                    os.path.join(diag_folder, f"epoch_{epoch + 1}_grad.npz"),
                    grad_row_norm=grad_arr,
                    grad_row_norm_mean=grad_arr.mean(axis=0),
                    grad_row_norm_std=grad_arr.std(axis=0),
                )

            # Save meta_records and ts2img_records to excel
            # Excel格式记录保存
            excel_folder = (
                self.args.results_folder
                + "train_results/"
                + setting
                + "/excel_records/"
            )
            os.makedirs(excel_folder, exist_ok=True)

            # 1. Save meta_records
            meta_list = [t.numpy() for t in model_ref.meta_records]
            meta_arr = (
                np.concatenate(meta_list, axis=0)
                if len(meta_list) > 0
                else np.array([])
            )

            if meta_arr.size > 0 and len(meta_arr.shape) == 3:
                meta_file_path = os.path.join(
                    excel_folder, f"epoch_{epoch + 1}_metaFeatures.xlsx"
                )
                with pd.ExcelWriter(meta_file_path) as writer:
                    # 每个变量一个sheet，每行一个样本的元特征
                    # meta_arr shape: [N, D, d_meta]
                    for d in range(meta_arr.shape[1]):
                        df = pd.DataFrame(meta_arr[:, d, :])
                        df.to_excel(writer, sheet_name=f"Variable_{d}", index=False)

            # 2. Save ts2img_records (w_arr)
            if w_arr.size > 0 and len(w_arr.shape) == 3:
                ts2img_file_path = os.path.join(
                    excel_folder, f"epoch_{epoch + 1}_ts2imgWeights.xlsx"
                )
                with pd.ExcelWriter(ts2img_file_path) as writer:
                    # w_arr shape: [N, D, d_ts2img]
                    for d in range(w_arr.shape[1]):
                        df = pd.DataFrame(w_arr[:, d, :])
                        df.to_excel(writer, sheet_name=f"Variable_{d}", index=False)

            model_ref.meta_records = []
            model_ref.ts2img_records = []
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
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
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

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
        preds = []
        trues = []
        folder_path = self.args.results_folder + "test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        # if hasattr(model_ref, "meta_mean"):
        #     print(f"test meta_mean: {model_ref.meta_mean}")
        # if hasattr(model_ref, "meta_std"):
        #     print(f"test meta_std: {model_ref.meta_std}")
        model_ref.meta_records = []
        model_ref.ts2img_records = []
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
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(
                        outputs.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                    batch_y = test_data.inverse_transform(
                        batch_y.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                preds.append(outputs)
                trues.append(batch_y)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(
                            input.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)
        folder_path = self.args.results_folder + "results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
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
            dtw_val = np.array(dtw_list).mean()
        else:
            dtw_val = "not calculated"
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse: {}, mae: {}, dtw: {}".format(mse, mae, dtw_val))
        results_path = self.args.results_path
        file_path = os.path.join("Result", results_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        f = open(file_path, "a")
        f.write(setting + "  \n")
        f.write("time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
        f.write("mse: {}, mae: {}, dtw: {}".format(mse, mae, dtw_val))
        f.write("\n")
        f.write("\n")
        f.close()
        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)
        meta_list = [t.numpy() for t in model_ref.meta_records]
        w_list = [t.numpy() for t in model_ref.ts2img_records]
        meta_arr = (
            np.concatenate(meta_list, axis=0) if len(meta_list) > 0 else np.array([])
        )
        w_arr = np.concatenate(w_list, axis=0) if len(w_list) > 0 else np.array([])
        np.save(folder_path + "meta_tensors.npy", meta_arr)
        np.save(folder_path + "ts2img_weights.npy", w_arr)
        return
