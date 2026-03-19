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
import json
from utils.dtw_metric import accelerated_dtw

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
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _model_core(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _get_router_aux_loss(self):
        return getattr(self._model_core(), "last_router_aux_loss", None)

    def _get_router_info(self):
        return getattr(self._model_core(), "last_router_info", {})

    def _empty_router_meter(self):
        return {
            "count": 0.0,
            "seg_weight_sum": 0.0,
            "smooth_weight_sum": 0.0,
            "seg_prefer_sum": 0.0,
            "smooth_prefer_sum": 0.0,
            "periodic_score_sum": 0.0,
            "trend_score_sum": 0.0,
        }

    def _update_router_meter(self, meter, batch_size):
        info = self._get_router_info()
        if not info:
            return
        meter["count"] += batch_size
        meter["seg_weight_sum"] += info.get("seg_weight_mean", 0.0) * batch_size
        meter["smooth_weight_sum"] += info.get("smooth_weight_mean", 0.0) * batch_size
        meter["seg_prefer_sum"] += info.get("seg_prefer_ratio", 0.0) * batch_size
        meter["smooth_prefer_sum"] += info.get("smooth_prefer_ratio", 0.0) * batch_size
        meter["periodic_score_sum"] += info.get("periodic_score_mean", 0.0) * batch_size
        meter["trend_score_sum"] += info.get("trend_score_mean", 0.0) * batch_size

    def _finalize_router_meter(self, meter):
        count = max(meter["count"], 1.0)
        return {
            "seg_weight_mean": meter["seg_weight_sum"] / count,
            "smooth_weight_mean": meter["smooth_weight_sum"] / count,
            "seg_prefer_ratio": meter["seg_prefer_sum"] / count,
            "smooth_prefer_ratio": meter["smooth_prefer_sum"] / count,
            "periodic_score_mean": meter["periodic_score_sum"] / count,
            "trend_score_mean": meter["trend_score_sum"] / count,
        }

    def _format_router_stats(self, stats):
        return (
            "seg_weight: {seg_weight_mean:.4f}, smooth_weight: {smooth_weight_mean:.4f}, "
            "seg_prefer: {seg_prefer_ratio:.4f}, smooth_prefer: {smooth_prefer_ratio:.4f}, "
            "periodic_score: {periodic_score_mean:.4f}, trend_score: {trend_score_mean:.4f}"
        ).format(**stats)

    def _metrics_dir(self, setting):
        folder_path = os.path.join(self.args.results_folder, "results", setting)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def _save_json(self, path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    def _unpack_batch(self, batch):
        if len(batch) == 4:
            return batch
        if len(batch) == 3:
            batch_x, batch_y, batch_x_mark = batch
            # fallback: use encoder marks for decoder marks when unavailable
            return batch_x, batch_y, batch_x_mark, batch_x_mark
        raise ValueError(f"Unexpected batch size: {len(batch)}")

    def _is_split_eval(self, loader_obj):
        return isinstance(loader_obj, list) and len(loader_obj) > 0 and isinstance(loader_obj[0], tuple) and len(loader_obj[0]) == 3

    def _vali_single_loader(self, data_set, data_loader, criterion):
        total_loss = []
        router_meter = self._empty_router_meter()

        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._unpack_batch(batch)
                batch_size = batch_x.shape[0]

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                self._update_router_meter(router_meter, batch_size)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                total_loss.append(criterion(pred, true).item())

        self.model.train()
        return np.average(total_loss), self._finalize_router_meter(router_meter)

    def vali(self, vali_data, vali_loader, criterion):
        if self._is_split_eval(vali_loader):
            per_dataset = []
            weighted_loss = 0.0
            weight_sum = 0.0
            combined_meter = self._empty_router_meter()

            for (name, ds, weight), (_, loader, _) in zip(vali_data, vali_loader):
                loss, stats = self._vali_single_loader(ds, loader, criterion)
                per_dataset.append({
                    "name": name,
                    "weight": float(weight),
                    "loss": float(loss),
                    "router": stats,
                })
                weighted_loss += float(weight) * float(loss)
                weight_sum += float(weight)

                # combine using dataset weight as pseudo-count
                combined_meter["count"] += float(weight)
                combined_meter["seg_weight_sum"] += stats["seg_weight_mean"] * float(weight)
                combined_meter["smooth_weight_sum"] += stats["smooth_weight_mean"] * float(weight)
                combined_meter["seg_prefer_sum"] += stats["seg_prefer_ratio"] * float(weight)
                combined_meter["smooth_prefer_sum"] += stats["smooth_prefer_ratio"] * float(weight)
                combined_meter["periodic_score_sum"] += stats["periodic_score_mean"] * float(weight)
                combined_meter["trend_score_sum"] += stats["trend_score_mean"] * float(weight)

            summary_stats = self._finalize_router_meter(combined_meter)
            return weighted_loss / max(weight_sum, 1e-12), summary_stats, per_dataset

        loss, stats = self._vali_single_loader(vali_data, vali_loader, criterion)
        return loss, stats, None

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        metrics_dir = self._metrics_dir(setting)
        epoch_router_records = []
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            if hasattr(train_data, "refresh_plan"):
                train_data.refresh_plan()

            iter_count = 0
            train_loss = []
            router_meter = self._empty_router_meter()

            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._unpack_batch(batch)
                batch_size = batch_x.shape[0]
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        self._update_router_meter(router_meter, batch_size)
                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y_cut = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y_cut)
                        router_aux = self._get_router_aux_loss()
                        if router_aux is not None:
                            loss = loss + router_aux
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    self._update_router_meter(router_meter, batch_size)
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y_cut = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y_cut)
                    router_aux = self._get_router_aux_loss()
                    if router_aux is not None:
                        loss = loss + router_aux

                train_loss.append(loss.item())

                if (i + 1) % 20 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
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
            train_router_stats = self._finalize_router_meter(router_meter)

            vali_loss, vali_router_stats, vali_detail = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_router_stats, test_detail = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss
            ))
            print("[Train Router] " + self._format_router_stats(train_router_stats))
            print("[Vali Router]  " + self._format_router_stats(vali_router_stats))
            print("[Test Router]  " + self._format_router_stats(test_router_stats))

            if vali_detail is not None:
                for item in vali_detail:
                    print("[Vali Split][{}][w={}] loss={:.7f} | {}".format(
                        item["name"], item["weight"], item["loss"], self._format_router_stats(item["router"])
                    ))
            if test_detail is not None:
                for item in test_detail:
                    print("[Test Split][{}][w={}] loss={:.7f} | {}".format(
                        item["name"], item["weight"], item["loss"], self._format_router_stats(item["router"])
                    ))

            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "vali_loss": float(vali_loss),
                "test_loss": float(test_loss),
                "train_router": train_router_stats,
                "vali_router": vali_router_stats,
                "test_router": test_router_stats,
                "vali_detail": vali_detail,
                "test_detail": test_detail,
            }
            epoch_router_records.append(epoch_record)
            self._save_json(os.path.join(metrics_dir, "router_epoch_metrics.json"), epoch_router_records)

            if vali_loss < self.best_val_loss:
                self.best_val_loss = vali_loss

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        print("best model save path: {}".format(best_model_path))
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def _test_single_loader(self, test_data, test_loader, setting, suffix=""):
        preds = []
        trues = []
        router_meter = self._empty_router_meter()

        folder_path = os.path.join(self.args.results_folder, "test_results", setting + suffix)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._unpack_batch(batch)
                batch_size = batch_x.shape[0]

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                self._update_router_meter(router_meter, batch_size)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                preds.append(outputs)
                trues.append(batch_y)

                if i % 20 == 0:
                    input_arr = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_arr.shape
                        input_arr = test_data.inverse_transform(input_arr.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input_arr[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input_arr[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

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
            dtw_value = float(np.array(dtw_list).mean())
        else:
            dtw_value = "not calculated"

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        router_stats = self._finalize_router_meter(router_meter)

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
            "mspe": float(mspe),
            "dtw": dtw_value,
            "router": router_stats,
            "preds": preds,
            "trues": trues,
        }

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        if test:
            print("loading model")
            path = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(path, "checkpoint.pth"), map_location=self.device))

        metrics_dir = self._metrics_dir(setting)

        if self._is_split_eval(test_loader):
            split_results = []
            weighted_mse = 0.0
            weighted_mae = 0.0
            weight_sum = 0.0

            for (name, ds, weight), (_, loader, _) in zip(test_data, test_loader):
                result = self._test_single_loader(ds, loader, setting, suffix="__" + name)
                result["name"] = name
                result["weight"] = float(weight)
                split_results.append(result)
                weighted_mse += float(weight) * result["mse"]
                weighted_mae += float(weight) * result["mae"]
                weight_sum += float(weight)

                np.save(os.path.join(metrics_dir, f"pred_{name}.npy"), result["preds"])
                np.save(os.path.join(metrics_dir, f"true_{name}.npy"), result["trues"])

                print("[Final Test Split][{}][w={}] mse: {}, mae: {}, dtw: {}".format(
                    name, weight, result["mse"], result["mae"], result["dtw"]
                ))
                print("[Final Test Split Router][{}] {}".format(
                    name, self._format_router_stats(result["router"])
                ))

            summary = {
                "weighted_mse": weighted_mse / max(weight_sum, 1e-12),
                "weighted_mae": weighted_mae / max(weight_sum, 1e-12),
                "splits": [
                    {
                        "name": r["name"],
                        "weight": r["weight"],
                        "mae": r["mae"],
                        "mse": r["mse"],
                        "rmse": r["rmse"],
                        "mape": r["mape"],
                        "mspe": r["mspe"],
                        "dtw": r["dtw"],
                        "router": r["router"],
                    }
                    for r in split_results
                ],
            }
            self._save_json(os.path.join(metrics_dir, "final_test_metrics.json"), summary)
            print("[Final Test Weighted] mse: {}, mae: {}".format(summary["weighted_mse"], summary["weighted_mae"]))
            return

        result = self._test_single_loader(test_data, test_loader, setting)
        print("mse: {}, mae: {}, dtw: {}".format(result["mse"], result["mae"], result["dtw"]))
        print("[Final Test Router] " + self._format_router_stats(result["router"]))

        test_metrics = {
            "mae": result["mae"],
            "mse": result["mse"],
            "rmse": result["rmse"],
            "mape": result["mape"],
            "mspe": result["mspe"],
            "dtw": result["dtw"],
            "router": result["router"],
        }
        self._save_json(os.path.join(metrics_dir, "final_test_metrics.json"), test_metrics)
        np.save(os.path.join(metrics_dir, "metrics.npy"), np.array([
            result["mae"], result["mse"], result["rmse"], result["mape"], result["mspe"]
        ]))
        np.save(os.path.join(metrics_dir, "pred.npy"), result["preds"])
        np.save(os.path.join(metrics_dir, "true.npy"), result["trues"])
        return
