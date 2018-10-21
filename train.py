from data import load_train_data, load_val_data
from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
from model import ResUNet
import numpy as np
import torch
import os
import torch.nn.functional as F
from metric import get_ave_precision
import sys
from time import gmtime, strftime


def criterion(logits, targets):
    logits = logits.squeeze(1)
    targets = targets.squeeze(1)
    loss = lovasz_hinge(logits, targets, per_image=True, ignore=None)
    return loss


def train(folder=0, weights_saving_path='weights', lr=0.001, weights=None, resuming_iter=0):
    net = ResUNet(pretrained=True)
    net.cuda()

    if weights is not None:
        print('Resume training from iter {:d}.'.format(resuming_iter))
        state_dict = torch.load(weights)['state_dicts']
        net.load_state_dict(state_dict)

    train_images, train_depths, train_masks = load_train_data(folder)
    val_images, val_depths, val_masks = load_val_data(folder)

    batch_size = 32
    one_epoch_iters = len(train_images) // batch_size
    saving_interval = one_epoch_iters // 2
    max_iters = one_epoch_iters * 500

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)

    bce_loss_layer = torch.nn.BCELoss()
    bce_loss_layer.cuda()

    if not os.path.exists(weights_saving_path):
        os.makedirs(weights_saving_path)

    print('batch_size = {:d}'.format(batch_size))
    print('one epoch iterations = {:d}'.format(one_epoch_iters))
    print('saving interval = {:d}'.format(saving_interval))
    print('max iterations = {:d} (~ {:d} epoches)'.format(max_iters, max_iters * batch_size // len(train_images)))
    print('lr = {:f}'.format(lr))

    time_str = strftime('%Y-%m-%d_%H-%M-%S', gmtime())
    file_name_suffix = '{:d}_{:s}'.format(folder, time_str)

    losses_file = open('losses_' + file_name_suffix + '.log', 'w')
    avg_losses_file = open('avg_losses_' + file_name_suffix + '.log', 'w')
    scores_file = open('scores_' + file_name_suffix + '.log', 'w')
    val_losses_file = open('val_losses_' + file_name_suffix + '.log', 'w')

    moving_avg_num = 50
    losses = []
    iter_idx = resuming_iter
    val_score = 0
    loss_changed = False

    if weights is not None:
        loss_name = 'bce'
        val_loss, val_score = validate(net, val_images, val_depths, val_masks, loss_name)

    while iter_idx < max_iters:
        # randomly pick one batch
        picked_samples = np.random.random_integers(0, len(train_images) - 1, batch_size)
        batch_images = train_images[picked_samples]
        batch_depths = train_depths[picked_samples]
        batch_masks = train_masks[picked_samples]

        input1 = torch.cuda.FloatTensor(batch_images)
        # input2 = torch.cuda.FloatTensor(batch_depths)

        targets = torch.cuda.FloatTensor(batch_masks)
        logits = net.forward(input1)

        if not loss_changed and (val_score < 0.81 and iter_idx <= 30 * one_epoch_iters):
            # if weights is None and iter_idx < first_stage_iters:
            loss = bce_loss_layer.forward(F.sigmoid(logits), targets)
            loss_name = 'bce'
        elif not loss_changed:
            loss = criterion(logits, targets)
            loss_name = 'lovasz'
            losses = []
            optimizer = torch.optim.Adam(net.parameters(), lr=lr *.5, weight_decay=1e-6)
            loss_changed = True
        else:
            # loss has already changed, so we don't care the val score		
            loss = criterion(logits, targets)
            loss_name = 'lovasz'

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_idx += 1

        loss = loss.item()
        losses.append(loss)
        moving_avg_loss = sum(losses[-moving_avg_num:])
        moving_avg_loss /= len(losses[-moving_avg_num:])
        losses = losses[-moving_avg_num:]

        losses_file.write('{:d} {:f}\n'.format(iter_idx, loss))
        avg_losses_file.write('{:d} {:f}\n'.format(iter_idx, moving_avg_loss))

        if iter_idx % saving_interval == 0:
            print('[Iter {:8d}] [{:s} loss {:8.4f}]  [avg_loss {:8.4f}] '.format(iter_idx, loss_name, loss, moving_avg_loss), end='', flush=False)
            val_loss, val_score = validate(net, val_images, val_depths, val_masks, loss_name)
            print('[val_{:s}_loss {:8.4f}] [val_score {:8.4f}]'.format(loss_name, val_loss, val_score))
            saving_name = os.path.join(weights_saving_path, 'iter_{:d}_loss_{:.4f}_score_{:.4f}.weights'.format(iter_idx, val_loss, val_score))
            torch.save({'state_dicts': net.state_dict()}, saving_name)

            val_losses_file.write('{:d} {:f}\n'.format(iter_idx, val_loss))
            scores_file.write('{:d} {:f}\n'.format(iter_idx, val_score))

        else:
            print('[Iter {:8d}] [{:s} loss {:8.4f}]  [avg_loss {:8.4f}]\r'.format(iter_idx, loss_name, loss, moving_avg_loss), end='', flush=False)

        if iter_idx % 20 == 0:
            # flush the log buffer, so we can plot the results in time.
            losses_file.flush()
            avg_losses_file.flush()
            scores_file.flush()
            val_losses_file.flush()


def validate(net, images, depths, masks, loss_name):
    avg_loss = 0
    # set to eval mode
    net.eval()

    predicted_masks = []
    target_masks = []

    for i in range(len(images)):
        image = images[i:i + 1, :, :, :]
        depth = depths[i:i + 1, :, :, :]
        mask = masks[i:i + 1, :, :, :]

        input1 = torch.cuda.FloatTensor(image)
        # input2 = torch.cuda.FloatTensor(depth)

        target = torch.cuda.FloatTensor(mask)
        logits = net.forward(input1)

        logits = logits.detach()
        pred = F.sigmoid(logits)

        if loss_name == 'bce':
            loss = F.binary_cross_entropy(pred, target)
        else:
            loss = criterion(logits, target)

        avg_loss += loss.item()

        pred = pred.squeeze()
        pred = pred.detach().cpu().numpy()
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0

        predicted_masks.append(pred)
        target_masks.append(mask.squeeze())

    score = get_ave_precision(predicted_masks, target_masks)
    avg_loss /= len(images)

    # back to training mode
    net.train()

    return avg_loss, score


if __name__ == '__main__':
    folder = int(sys.argv[1])
    weights_saving_path = sys.argv[2]

    if len(sys.argv) >= 4:
        lr = float(sys.argv[3])
    else:
        lr = 0.001

    if len(sys.argv) >= 5:
        weights_path = sys.argv[4]
    else:
        weights_path = None

    if len(sys.argv) >= 6:
        resuming_iter = int(sys.argv[5])
    else:
        resuming_iter = 0

    train(folder, weights_saving_path, lr, weights_path, resuming_iter)
