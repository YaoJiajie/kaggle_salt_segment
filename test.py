from model import ResUNet
from data import load_test_data
import torch
import numpy as np
import torch.nn.functional as F
import sys
import cv2
import os


def run_length_encoding(mask):
    mask = np.transpose(mask)
    ys, xs, = np.nonzero(mask)
    height, width = mask.shape[0], mask.shape[1]
    code = []
    beg_idx = -1
    last_idx = -1
    count = 0

    for y, x in zip(ys, xs):
        idx = y * width + x + 1
        if idx == last_idx + 1:
            last_idx = idx
            count += 1
        else:
            if count > 0:
                # finish the last round
                code.append(str(beg_idx))
                code.append(str(count))

            # begin new round
            beg_idx = idx
            count = 1
            last_idx = idx

    # last unfinished round
    if count > 0:
        code.append(str(beg_idx))
        code.append(str(count))

    # convert to string
    code = ' '.join(code)
    return code


def test(weights_paths):
    nets = []
    for weights_path in weights_paths:
        net = ResUNet()
        net.cuda()
        state_dict = torch.load(weights_path)['state_dicts']
        net.load_state_dict(state_dict)
        net.eval()
        nets.append(net)

    images, image_ids, image_sizes, depths = load_test_data()
    total_count = len(images)
    pred_masks = []

    print('Predicting ...')

    for i, (img, img_size, depth) in enumerate(zip(images, image_sizes, depths)):
        img = img[np.newaxis, :, :, :]
        depth_blob = np.zeros_like(img)
        depth_blob[:] = depth

        h, w = img_size
        input1 = torch.cuda.FloatTensor(img)
        # input2 = torch.cuda.FloatTensor(depth_blob)

        combined_logits = None
        for net in nets:
            # logits = net.forward(input1, input2)
            logits = net.forward(input1)

            # pred = F.sigmoid(logits)
            # pred = pred.detach().cpu().numpy()
            # pred = pred.squeeze()

            if combined_logits is None:
                combined_logits = logits
            else:
                combined_logits += logits

        combined_logits /= len(nets)
        pred = F.sigmoid(combined_logits)
        pred = pred.detach().cpu().numpy()
        pred = pred.squeeze()

        thresh = 0.42
        pred[pred > thresh] = 1
        pred[pred <= thresh] = 0
        padded_h, padded_w = pred.shape
        off_h = (padded_h - h) // 2
        off_w = (padded_w - w) // 2
        pred = np.copy(pred[off_h:off_h + h, off_w:off_w + w])
        pred_masks.append(pred)
        print('{:6d} / {:6d}\r'.format(i + 1, total_count), end='', flush=True)

    print('')
    print('Writing to file...')
    result_file = open('result.csv', 'w')
    result_file.write('id,rle_mask\n')

    empty_count = 0
    count = 0
    for mask, image_id in zip(pred_masks, image_ids):
        mask = mask.astype(np.uint8)
        
        # if np.count_nonzero(mask) <= 50:
        #     mask[:] = 0
        
        if np.count_nonzero(mask) == 0:
            empty_count += 1

        code = run_length_encoding(mask)
        result_file.write('{:s},{:s}\n'.format(image_id, code))
        count += 1
        print('{:8d}/{:8d}\r'.format(count, total_count), end='', flush=True)

    print('')
    print('Test Done, empty ratio = ({:d}/{:d})'.format(empty_count, len(pred_masks)))


def test_with_tta(weights_paths):
    nets = []
    for weights_path in weights_paths:
        net = ResUNet()
        net.cuda()
        state_dict = torch.load(weights_path)['state_dicts']
        net.load_state_dict(state_dict)
        net.eval()
        nets.append(net)
        # break

    images, image_ids, image_sizes, depths = load_test_data()
    total_count = len(images)

    if not os.path.exists('raw_preds.npy'):
        raw_preds = []
        print('Predicting ...')

        for i, (img, img_size, depth) in enumerate(zip(images, image_sizes, depths)):
            img = img[np.newaxis, :, :, :]
            depth_blob = np.zeros_like(img)
            depth_blob[:] = depth

            padded_h, padded_w = img.shape[2], img.shape[3]
            h, w = img_size

            off_h = (padded_h - h) // 2
            off_w = (padded_w - w) // 2

            img_croped = img[0, 0, off_h:off_h + h, off_w:off_w + w]
            img_rescaled = cv2.resize(img_croped, (padded_w, padded_h))
            img_rescaled = img_rescaled[np.newaxis, np.newaxis, :, :]

            input_extended = torch.cuda.FloatTensor(img)
            input_rescaled = torch.cuda.FloatTensor(img_rescaled)

            combined_logits_extened = None
            combined_logits_scaled = None

            for net in nets:
                logits = net.forward(input_extended)

                if combined_logits_extened is None:
                    combined_logits_extened = logits
                else:
                    combined_logits_extened += logits

                logits = net.forward(input_rescaled)
                if combined_logits_scaled is None:
                    combined_logits_scaled = logits
                else:
                    combined_logits_scaled += logits

            combined_logits_extened /= len(nets)
            combined_logits_scaled /= len(nets)

            pred_extended = combined_logits_extened.detach()
            pred_scaled = combined_logits_scaled.detach()

            pred_extended = pred_extended.squeeze()
            pred_extended = pred_extended[off_h:off_h + h, off_w:off_w + w]

            pred_scaled = F.interpolate(pred_scaled, (h, w))
            pred_scaled = pred_scaled.squeeze()

            pred = (pred_scaled + pred_extended) / 2
            pred = F.sigmoid(pred)

            pred = pred.detach().cpu().numpy()
            raw_preds.append(pred)
            print('{:6d} / {:6d}\r'.format(i + 1, total_count), end='', flush=True)

        raw_preds = np.array(raw_preds)
        np.save('raw_preds.npy', raw_preds)

    pred_masks = []
    raw_preds = np.load('raw_preds.npy')
    thresh = 0.40
    for pred in raw_preds:
        pred[pred > thresh] = 1
        pred[pred <= thresh] = 0
        pred_masks.append(pred)

    print('')
    print('Writing to file...')
    result_file = open('result.csv', 'w')
    result_file.write('id,rle_mask\n')

    empty_count = 0
    count = 0
    for mask, image_id in zip(pred_masks, image_ids):
        mask = mask.astype(np.uint8)

        if np.count_nonzero(mask) == 0:
            empty_count += 1

        code = run_length_encoding(mask)
        result_file.write('{:s},{:s}\n'.format(image_id, code))
        count += 1
        print('{:8d} / {:8d}\r'.format(count, total_count), end='', flush=True)

    print('')
    print('Test Done, empty ratio = ({:d}/{:d})'.format(empty_count, len(pred_masks)))


if __name__ == '__main__':
    # test(sys.argv[1:])
    test_with_tta(sys.argv[1:])
