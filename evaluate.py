from model import ResUNet
from data import load_val_data
import torch
import torch.nn.functional as F
from metric import get_ave_precision
import sys
import numpy as np


def evaluate(net, images, masks):
    predicted_masks = []
    masks = masks.squeeze(1)

    for i in range(len(images)):
        image = images[i:i + 1, :, :, :]
        input = torch.cuda.FloatTensor(image)
        logits = net.forward(input)

        logits = logits.detach()
        predict = F.sigmoid(logits)
        predict = predict.cpu().numpy()

        predict = predict.squeeze()
        thresh = 0.40
        predict[predict > thresh] = 1
        predict[predict <= thresh] = 0

        # if np.count_nonzero(predict) <= 50:
        #    predict[:] = 0

        predicted_masks.append(predict)
        print('( {:6d} / {:6d} )\r'.format(i + 1, len(images)), end='', flush=True)

    print('')
    score = get_ave_precision(predicted_masks, masks)
    print('Score = {:f}'.format(score))


if __name__ == '__main__':

    net = ResUNet(pretrained=False)
    state_dict = torch.load(sys.argv[1])['state_dicts']
    net.load_state_dict(state_dict)
    net.cuda()
    net.eval()

    folder = 0
    if len(sys.argv) >= 3:
        folder = int(sys.argv[2])

    images, _, masks = load_val_data(folder)
    evaluate(net, images, masks)
