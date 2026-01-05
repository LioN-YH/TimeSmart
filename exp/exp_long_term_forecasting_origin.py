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

        # 初始化用于 Optuna 的监控指标
        # 设为无穷大，表示尚未训练
        self.best_val_loss = float("inf")

    # 构建模型
    def _build_model(self):
        # 根据传入的参数选择并初始化一个模型
        model = self.model_dict[self.args.model].Model(self.args).float()

        # 多 GPU 并行训练
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # 获取数据集和加载器
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # 选择优化器
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # 选择损失函数
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    # 验证过程
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        # 将模型设置为评估模式
        self.model.eval()
        # 临时禁用梯度计算
        with torch.no_grad():
            # enumerate过程中实际上是dataloader按照其参数规定的策略调用了其dataset的__getitem__方法
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                # 1 数据预处理
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 2 构建解码器输入【label_len：拷贝，pred_len：零向量】
                # decoder input
                # 已知目标序列batch_y数据包含两部分：已知部分label_len和预测部分pred_len
                # dec_inp是一个与目标序列后 pred_len个时间步形状相同的零张量
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                # 将目标序列的前 label_len 个时间步与 dec_inp 拼接，形成完整的解码器输入
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                # 3 前向传播
                # encoder - decoder
                # 启用混合精度训练 (use_amp)，则使用autocast上下文管理器进行前向传播，以减少内存占用并加速计算
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                # 否则直接调用模型进行前向传播
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # 4 调整输出和目标的维度
                # f_dim = -1 表示取最后一个特征维度（即目标变量所在的列）
                # f_dim = 0  表示取所有特征维度
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                # 5 计算损失
                # 将预测值和真实值移回 CPU，并分离出计算图
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        # 恢复模型为训练模式
        self.model.train()
        # 返回平均验证损失
        return total_loss

    # 训练过程
    def train(self, setting):
        # 1 数据准备
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        # 2 模型保存路径设置
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # 3 初始化训练相关变量
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 4 混合精度训练支持：创建 GradScaler 用于缩放梯度，防止数值下溢
        # == autocast与GradScaler一起使用 ==：
        # 因为autocast会损失部分精度，从而导致梯度消失的问题，并且经过中间层时可能计算得到inf导致最终loss出现nan
        # 所以通常将GradScaler与autocast配合使用来对梯度值进行一些放缩，来缓解上述的一些问题
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # 5 开始训练循环
        for epoch in range(self.args.train_epochs):
            # [1] 初始每个epoch的变量
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # [2] 遍历训练数据（小批量训练）
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                # 梯度清零
                model_optim.zero_grad()
                # 数据预处理与设备转移
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 构建解码器输入
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                # 前向传播&损失计算
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

                # 打印训练进度：当前损失、速度和预计剩余时间
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

                # 梯度更新（反向传播）
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # [3] 每个epoch结束后，打印训练损失和验证损失
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )

            # 更新最佳验证损失，用于 Optuna 监控
            if vali_loss < self.best_val_loss:
                self.best_val_loss = vali_loss

            # [4] 使用验证损失判断是否触发早停机制
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # [5] 调整学习率
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 6 训练结束后，加载最佳模型，并返回训练好的模型
        best_model_path = path + "/" + "checkpoint.pth"
        print("best model save path: {}".format(best_model_path))
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    # 测试过程
    def test(self, setting, test=0):
        # 1 获取测试数据
        test_data, test_loader = self._get_data(flag="test")

        # test: 控制是否从检查点加载最优模型进行测试（1 表示加载，0 表示使用当前模型）
        if test:
            print("loading model")
            path = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(
                torch.load(
                    os.path.join(path, "checkpoint.pth"),
                    map_location=self.device,
                )
            )

        preds = []
        trues = []
        folder_path = self.args.results_folder + "test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 2 测试过程
        # 以下代码写法与验证过程基本相同
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

                # 反归一化
                # 训练时对数据进行了标准化（如 MinMaxScaler），这里将预测值和真实值还原回原始尺度
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    # 展平张量以便传入inverse_transform
                    outputs = test_data.inverse_transform(
                        outputs.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                    batch_y = test_data.inverse_transform(
                        batch_y.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)

                # 特征维度筛选
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                # 可视化（每20个batch显示一次）
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(
                            input.reshape(shape[0] * shape[1], -1)
                        ).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        # 3 合并所有batch的预测和真实值
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        # result save
        folder_path = self.args.results_folder + "results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 4 DTW计算（可选）
        # DTW是一种衡量两个时间序列相似性的方法，特别适用于长度不一致或存在偏移的情况
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
        print("mse: {}, mae: {}, dtw: {}".format(mse, mae, dtw))
        # 5 写入文本日志
        results_path = self.args.results_path
        file_path = os.path.join("Result", results_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        f = open(file_path, "a")
        f.write(setting + "  \n")
        f.write("mse: {}, mae: {}, dtw: {}".format(mse, mae, dtw))
        f.write("\n")
        f.write("\n")
        f.close()

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)

        return
