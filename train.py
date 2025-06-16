import torch.nn
from torch.autograd import Variable
# import numpy as np
import os, argparse
from datetime import datetime
from new_SOD_pvt_gai_merge_decoder_SOTA import *

from utils.data import get_loader, test_dataset
from utils.func import label_edge_prediction, AvgMeter, clip_gradient, adjust_lr
import pytorch_iou
import pytorch_fm

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=80, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352,
                    help='training dataset size')  # The trainsize of SEINet_ResNet50 and SEINet2_ResNet50 is 352
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
opt = parser.parse_args()


def joint_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


print('Learning Rate: {}'.format(opt.lr))
# build models
model = SOD()

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

image_root = r'.\train\image/'
gt_root = r'F.\train\GT/'
val_image_root = r'.\test\image/'
val_gt_root = r'.\test\GT/'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(val_image_root, val_gt_root, opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average=True)
floss = pytorch_fm.FLoss()
sCE = torch.nn.BCELoss()
size_rates = [0.75, 1, 1.25]  # multi-scale training


def train(train_loader, model, optimizer, epoch):
    model.train()
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # edge prediction
            edges = label_edge_prediction(gts)

            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)

            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                edges = F.interpolate(edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            o1, o2, o3, o4, edg_ls = model(images)
            # bce+iou+fmloss
            loss1 = CE(o1, gts) + IOU(F.sigmoid(o1), gts)
            loss2 = CE(o2, gts) + IOU(F.sigmoid(o2), gts)
            loss3 = CE(o3, gts) + IOU(F.sigmoid(o3), gts)
            loss4 = CE(o4, gts) + IOU(F.sigmoid(o4), gts)
            loss5 = sCE(edg_ls, edges)

            loss = loss1 + loss2 + loss3 + loss4 + loss5

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            if rate == 1:
                loss_record1.update(loss1.data, opt.batchsize)
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)

        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f},  Loss_edge: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data,
                           loss5.data))

    # save_path = './models/earth/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # if (epoch + 1) >= 10:
    #     torch.save(model.state_dict(), save_path + 'Our_earth.pth' + '.%d' % epoch)


def val(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            pre_res = model(image)
            res = pre_res[0]
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        # writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        # print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch{}_earth_{}.pth'.format(epoch, best_mae))
                print('best epoch:{}'.format(epoch))
        print(' MAE: {} ####  bestMAE: {} '.format(mae, best_mae))


print("Let's go!")
if __name__ == '__main__':
    save_path = r'./models\MYnet2/'
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
        val(test_loader, model, epoch, save_path)
