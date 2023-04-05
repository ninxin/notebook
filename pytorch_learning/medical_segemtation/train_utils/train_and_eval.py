import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    # print(type(inputs))
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        # print(x.shape)
        # print(target.shape)
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)    # 计算损失时不计算忽略的
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


# # AGNet
# def calculate_Accuracy(confusion):
#     confusion = np.asarray(confusion)
#     pos = np.sum(confusion, 1).astype(np.float32)  # 1 for row
#     res = np.sum(confusion, 0).astype(np.float32)  # 0 for coloum
#     tp = np.diag(confusion).astype(np.float32)
#     IU = tp / (pos + res - tp)
#
#     meanIU = np.mean(IU)
#     Acc = np.sum(tp) / np.sum(confusion)
#     Se = confusion[1][1] / (confusion[1][1] + confusion[0][1])
#     Sp = confusion[0][0] / (confusion[0][0] + confusion[1][0])
#
#     return meanIU, Acc, Se, Sp, IU
#
#
# Background_IOU = []
# Vessel_IOU = []
# ACC = []
# SE = []
# SP = []
# AUC = []
#
#
# # AGNet


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            # Iternet
            # output = output[3]['out']

            # # AGNet
            # output, side_5, side_6, side_7, side_8 = model(image)
            # output = side_8["out"]
            # # output = softmax_2d(side_8["out"]) + EPS
            # # print(side_8["out"].shape)
            # # # output = nn.Softmax2d(side_8["out"])
            # # # print(output.shape)
            # #
            # # output = output.cpu().data.numpy()
            # # print(np.unique(output))
            # # y_pred = output[:, 1, :, :]
            # # y_pred = y_pred.reshape([-1])
            # # ppi = np.argmax(output, 1)
            # #
            # # tmp_out = ppi.reshape([-1])
            # # tmp_gt = target.cpu().reshape([-1])
            # #
            # # my_confusion = metrics.confusion_matrix(tmp_out, tmp_gt).astype(np.float32)
            # # meanIU, Acc, Se, Sp, IU = calculate_Accuracy(my_confusion)
            # # Auc = roc_auc_score(tmp_gt, y_pred)
            # # AUC.append(Auc)
            # #
            # # Background_IOU.append(IU[0])
            # # Vessel_IOU.append(IU[1])
            # # ACC.append(Acc)
            # # SE.append(Se)
            # # SP.append(Sp)
            #
            # # AGNet

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()
        # print('Acc: %s  |  Se: %s |  Sp: %s |  Auc: %s |  Background_IOU: %s |  vessel_IOU: %s ' % (
        #     str(np.mean(np.stack(ACC))), str(np.mean(np.stack(SE))), str(np.mean(np.stack(SP))),
        #     str(np.mean(np.stack(AUC))),
        #     str(np.mean(np.stack(Background_IOU))), str(np.mean(np.stack(Vessel_IOU)))))

    return confmat, dice.value.item()
    # return {"ACC": np.mean(np.stack(ACC)), "Se": np.mean(np.stack(SE)), "Sp": np.mean(np.stack(SP)),
    #         "Auc": np.mean(np.stack(AUC)),
    #         "Back_IOU": np.mean(np.stack(Background_IOU)),
    #         "vessel_IOU": np.mean(np.stack(Vessel_IOU))}, dice.value.item()


# # AGNet
# EPS = 1e-12
# # criterion = nn.NLLLoss()
# softmax_2d = nn.Softmax2d()
# # def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
# #     losses = {}
# #     # print(type(inputs))
# #     print(inputs.shape)
# #     for x in inputs:
# #         # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
# #         print(x.shape)
# #         print(target.shape)
# #         loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)    # 计算损失时不计算忽略的
# #         if dice is True:
# #             dice_target = build_target(target, num_classes, ignore_index)
# #             loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
# #         losses['out'] = loss
# #
# #     if len(losses) == 1:
# #         return losses['out']
# #
# #     return losses['out'] + 0.5 * losses['aux']
#
#
# # AGNet

def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            # print(output.shape)
            # print(target.shape)
            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)

            # # # Iternet
            # loss1 = criterion(output[0], target, loss_weight, num_classes=num_classes, ignore_index=255)
            # loss2 = criterion(output[1], target, loss_weight, num_classes=num_classes, ignore_index=255)
            # loss3 = criterion(output[2], target, loss_weight, num_classes=num_classes, ignore_index=255)
            # loss4 = criterion(output[3], target, loss_weight, num_classes=num_classes, ignore_index=255)
            # loss = (loss1+loss2+loss3+loss4)/4

            # # AGNet
            # out, side_5, side_6, side_7, side_8 = model(image)
            # # print(out.shape)
            # out['out'] = torch.log(softmax_2d(out['out']) + EPS)
            # side_5['out'] = torch.log(softmax_2d(side_5['out']) + EPS)
            # side_6['out'] = torch.log(softmax_2d(side_6['out']) + EPS)
            # side_7['out'] = torch.log(softmax_2d(side_7['out']) + EPS)
            # side_8['out'] = torch.log(softmax_2d(side_8['out']) + EPS)
            # # print(out.shape)
            # # loss = criterion(out, target)
            # # loss += criterion(torch.log(softmax_2d(side_5) + EPS), target)
            # # loss += criterion(torch.log(softmax_2d(side_6) + EPS), target)
            # # loss += criterion(torch.log(softmax_2d(side_7) + EPS), target)
            # # loss += criterion(torch.log(softmax_2d(side_8) + EPS), target)
            # loss = criterion(out, target, loss_weight, num_classes=num_classes, ignore_index=255)
            # loss += criterion(side_5, target, loss_weight, num_classes=num_classes, ignore_index=255)
            # loss += criterion(side_6, target, loss_weight, num_classes=num_classes, ignore_index=255)
            # loss += criterion(side_7, target, loss_weight, num_classes=num_classes, ignore_index=255)
            # loss += criterion(side_8, target, loss_weight, num_classes=num_classes, ignore_index=255)
            # # out = torch.log(softmax_2d(side_8) + EPS)
            # # AGNet

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
