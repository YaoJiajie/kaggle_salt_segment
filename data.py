import os
import pandas as pd
import cv2
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt


raw_data_root = os.path.join(str(Path.home()), '.kaggle/competitions/tgs-salt-identification-challenge')
npy_root = 'npy_data'
test_images_path = os.path.join(raw_data_root, 'test', 'images')
input_height = 128
input_width = 128


def mirror_padding(image, target_height, target_width):
    height, width = image.shape[0], image.shape[1]
    assert height <= target_height and width <= target_width

    result = np.zeros((target_height, target_width), np.uint8)
    height_off_top = (target_height - height) // 2
    width_off_left = (target_width - width) // 2
    height_off_bot = (target_height - height) - height_off_top
    width_off_right = (target_width - width) - width_off_left

    result[height_off_top:height_off_top + height, width_off_left:width_off_left + width] = image

    # four boundaries:
    result[:height_off_top, width_off_left:width_off_left + width] = image[:height_off_top, :][::-1, :]
    result[height_off_top:height_off_top + height, :width_off_left] = image[:, :width_off_left][:, ::-1]

    result[target_height - height_off_bot:, width_off_left:width_off_left + width] = image[height - height_off_bot:, :][::-1, :]
    result[height_off_top:height_off_top + height, target_width - width_off_right:] = image[:, width - width_off_right:][:, ::-1]

    # cv2.imshow('result', result)
    # cv2.waitKey()
    return result


def flip(image, axis):
    new_image = image.copy()

    # up/down flip
    if axis == 0:
        new_image[:, :] = image[::-1, :]
    # left/right flip
    else:
        new_image[:, :] = image[:, ::-1]
    return new_image


def rotate(image, degree):
    rows, cols = image.shape[0], image.shape[1]
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), degree, 1)
    dst = cv2.warpAffine(image, M, (cols, rows))
    return dst


def crop_and_scale(image, roi):
    h, w = image.shape[0], image.shape[1]
    roi_image = image[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    rescaled_roi_image = cv2.resize(roi_image, (w, h))
    return rescaled_roi_image


def make_crop_fn(h_ratio, w_ratio):
    def _crop_fn(image, mask):
        h, w = image.shape[0], image.shape[1]

        roi_h = int(h * h_ratio)
        roi_w = int(w * w_ratio)
        roi_x = random.randint(0, w - roi_w)
        roi_y = random.randint(0, h - roi_h)
        roi = (roi_x, roi_y, roi_w, roi_h)

        new_image = crop_and_scale(image, roi)
        new_mask = crop_and_scale(mask, roi)
        return new_image, new_mask
    return _crop_fn


def prepare_train_data():

    if not os.path.exists(npy_root):
        os.makedirs(npy_root)

    train_csv_path = os.path.join(raw_data_root, 'train.csv')
    train_csv = pd.read_csv(train_csv_path)
    depth_dict = load_depth_values()

    raw_images = []
    raw_masks = []
    depth_values = []

    mirror_extended_raw_images = []
    mirror_extended_raw_masks = []

    for _, row in train_csv.iterrows():
        image_id = row['id']
        image = cv2.imread(os.path.join(raw_data_root, 'images', image_id + '.png'), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(raw_data_root, 'masks', image_id + '.png'), cv2.IMREAD_GRAYSCALE)
        mask[mask != 0] = 1

        depth_values.append(depth_dict[image_id])
        raw_images.append(image)
        raw_masks.append(mask)
    print('total train num = {:d}'.format(len(raw_images)))

    for image, mask in zip(raw_images, raw_masks):
        image = mirror_padding(image, input_height, input_width)
        mask = mirror_padding(mask, input_height, input_width)
        mirror_extended_raw_images.append(image)
        mirror_extended_raw_masks.append(mask)

    raw_images_array = np.array(raw_images)
    raw_masks_array = np.array(raw_masks)
    depth_values_array = np.array(depth_values)

    np.save(os.path.join(npy_root, 'raw_depths.npy'), depth_values_array)
    np.save(os.path.join(npy_root, 'raw_images.npy'), raw_images_array)
    np.save(os.path.join(npy_root, 'raw_masks.npy'), raw_masks_array)

    extended_raw_images_array = np.array(mirror_extended_raw_images)
    extended_raw_masks_array = np.array(mirror_extended_raw_masks)
    np.save(os.path.join(npy_root, 'extended_raw_images.npy'), extended_raw_images_array)
    np.save(os.path.join(npy_root, 'extended_raw_masks.npy'), extended_raw_masks_array)

    # split train/val for local evaluation
    mirror_extended_raw_images_masks = list(zip(mirror_extended_raw_images, mirror_extended_raw_masks, depth_values))
    random.shuffle(mirror_extended_raw_images_masks)
    val_images_num = len(mirror_extended_raw_images_masks) // 6

    images, masks, depths = zip(*mirror_extended_raw_images_masks)
    val_images = images[:val_images_num]
    val_masks = masks[:val_images_num]
    val_depths = depths[:val_images_num]

    val_images = np.array(val_images)
    val_masks = np.array(val_masks)
    val_depths = np.array(val_depths)

    np.save(os.path.join(npy_root, 'val_images.npy'), val_images)
    np.save(os.path.join(npy_root, 'val_masks.npy'), val_masks)
    np.save(os.path.join(npy_root, 'val_depths.npy'), val_depths)

    print('Split out {:d} samples for validation.'.format(len(val_images)))

    # make augmentation (normal + flip*2 +  random crop & rescale * 3)
    train_images = images[val_images_num:]
    train_masks = masks[val_images_num:]
    train_depths = depths[val_images_num:]

    crop1 = make_crop_fn(0.8, 0.8)
    crop2 = make_crop_fn(0.6, 0.6)
    crop3 = make_crop_fn(0.9, 0.9)
    crops = [crop1, crop2, crop3]

    train_aug_images = []
    train_aug_masks = []
    train_aug_depths = []

    for image, mask, depth in zip(train_images, train_masks, train_depths):
        train_aug_images.append(image)
        train_aug_masks.append(mask)
        train_aug_depths.append(depth)

        train_aug_images.append(flip(image, 0))
        train_aug_masks.append(flip(mask, 0))
        train_aug_depths.append(depth)

        train_aug_images.append(flip(image, 1))
        train_aug_masks.append(flip(mask, 1))
        train_aug_depths.append(depth)

        train_aug_images.append(rotate(image, 45))
        train_aug_masks.append(rotate(mask, 45))
        train_aug_depths.append(depth)

        train_aug_images.append(rotate(image, -60))
        train_aug_masks.append(rotate(mask, -60))
        train_aug_depths.append(depth)
        
        train_aug_images.append(rotate(image, 30))
        train_aug_masks.append(rotate(mask, 30))
        train_aug_depths.append(depth)
        
        for crop in crops:

            crop_img, crop_mask = crop(image, mask)
            train_aug_images.append(crop_img)
            train_aug_masks.append(crop_mask)
            train_aug_depths.append(depth)

            # cv2.imshow('crop_img', crop_img)
            # cv2.imshow('crop_mask', crop_mask * 255)
            # cv2.waitKey()

    train_aug_images = np.array(train_aug_images)
    train_aug_masks = np.array(train_aug_masks)
    train_aug_depths = np.array(train_aug_depths)

    np.save(os.path.join(npy_root, 'train_images.npy'), train_aug_images)
    np.save(os.path.join(npy_root, 'train_masks.npy'), train_aug_masks)
    np.save(os.path.join(npy_root, 'train_depths.npy'), train_aug_depths)

    print('Generate {:d} train samples.'.format(len(train_aug_images)))


def prepare_test_data():
    if not os.path.exists(npy_root):
        os.makedirs(npy_root)

    images = []
    image_ids = []
    image_sizes = []
    depths = []
    depth_dict = load_depth_values()

    for image_name in os.listdir(test_images_path):
        image_id, _ = os.path.splitext(image_name)
        image = cv2.imread(os.path.join(test_images_path, image_name), cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        image = mirror_padding(image, input_height, input_width)

        images.append(image)
        image_ids.append(image_id)
        image_sizes.append((w, h))
        depths.append(depth_dict[image_id])

    images = np.array(images)
    image_sizes = np.array(image_sizes)
    depths = np.array(depths)

    np.save(os.path.join(npy_root, 'test_images.npy'), images)
    np.save(os.path.join(npy_root, 'test_image_ids.npy'), image_ids)
    np.save(os.path.join(npy_root, 'test_image_sizes.npy'), image_sizes)
    np.save(os.path.join(npy_root, 'test_depths.npy'), depths)
    print('Load {:d} test images.'.format(len(images)))


def load_data(images_name, depths_name, masks_name=None):
    images = np.load(os.path.join(npy_root, images_name))
    images = images.astype(np.float32)
    images = images[:, np.newaxis, :, :]
    images /= 255.0

    depths = np.load(os.path.join(npy_root, depths_name))
    depths = depths.astype(np.float32)
    depths = depths[:, np.newaxis, np.newaxis, np.newaxis]
    depths_blob = np.copy(images)
    depths_blob[:] = depths

    if masks_name is not None:
        masks = np.load(os.path.join(npy_root, masks_name))
        masks = masks.astype(np.float32)
        masks = masks[:, np.newaxis, :, :]
        return images, depths_blob, masks

    return images, depths_blob


def load_train_data(folder=0):
    if folder == 0:
        return load_data('train_images.npy', 'train_depths.npy', 'train_masks.npy')
    else:
        train_images_name = 'train_images_folder_{:d}.npy'.format(folder)
        train_masks_name = 'train_masks_folder_{:d}.npy'.format(folder)
        train_depths_name = 'train_depths_folder_{:d}.npy'.format(folder)
        return load_data(train_images_name, train_depths_name, train_masks_name)


def load_val_data(folder=0):
    if folder == 0:
        return load_data('val_images.npy', 'val_depths.npy', 'val_masks.npy')
    else:
        val_images_name = 'val_images_folder_{:d}.npy'.format(folder)
        val_masks_name = 'val_masks_folder_{:d}.npy'.format(folder)
        val_depths_name = 'val_depths_folder_{:d}.npy'.format(folder)
        return load_data(val_images_name, val_depths_name, val_masks_name)


def load_test_data():
    images, depths = load_data('test_images.npy', 'test_depths.npy')
    image_ids = np.load(os.path.join(npy_root, 'test_image_ids.npy'))
    image_sizes = np.load(os.path.join(npy_root, 'test_image_sizes.npy'))
    # depths = np.load(os.path.join(npy_root, 'test_depths.npy'))
    return images, image_ids, image_sizes, depths


def create_n_folder(n):

    images = np.load(os.path.join(npy_root, 'extended_raw_images.npy'))
    masks = np.load(os.path.join(npy_root, 'extended_raw_masks.npy'))
    depths = np.load(os.path.join(npy_root, 'raw_depths.npy'))

    print('load {:d} images.'.format(len(images)))

    image_mask_depth_pairs = list(zip(images, masks, depths))
    random.shuffle(image_mask_depth_pairs)

    total_num = len(image_mask_depth_pairs)
    intervals = np.linspace(0, total_num, n + 1)
    intervals = intervals.astype(np.int32)

    sections = []
    for i in range(n):
        beg = intervals[i]
        end = intervals[i + 1]
        pairs = image_mask_depth_pairs[beg:end]
        sections.append(pairs)

    # each time, we pick one section as val set.
    # remaining as train set.
    for i in range(n):
        val_set = sections[i]
        train_set = []
        for j in range(n):
            if j != i:
                train_set += sections[j]

        val_images, val_masks, val_depths = zip(*val_set)
        train_images, train_masks, train_depths = zip(*train_set)

        print('folder {:d}, has {:d} val images, {:d} train images.'.format(i + 1, len(val_images), len(train_images)))

        val_images = np.array(val_images)
        val_masks = np.array(val_masks)
        val_depths = np.array(val_depths)
        np.save(os.path.join(npy_root, 'val_images_folder_{:d}.npy'.format(i + 1)), val_images)
        np.save(os.path.join(npy_root, 'val_masks_folder_{:d}.npy'.format(i + 1)), val_masks)
        np.save(os.path.join(npy_root, 'val_depths_folder_{:d}.npy'.format(i + 1)), val_depths)

        # apply augmentation (original, crop & rescale * 3, flip * 2)
        train_aug_images = []
        train_aug_masks = []
        train_aug_depths = []

        crop1 = make_crop_fn(0.8, 0.8)
        crop2 = make_crop_fn(0.6, 0.6)
        crop3 = make_crop_fn(0.9, 0.9)
        crops = [crop1, crop2, crop3]

        for image, mask, depth in zip(train_images, train_masks, train_depths):
            train_aug_images.append(image)
            train_aug_masks.append(mask)
            train_aug_depths.append(depth)

            train_aug_images.append(flip(image, 0))
            train_aug_masks.append(flip(mask, 0))
            train_aug_depths.append(depth)

            train_aug_images.append(flip(image, 1))
            train_aug_masks.append(flip(mask, 1))
            train_aug_depths.append(depth)
            
            train_aug_images.append(rotate(image, 45))
            train_aug_masks.append(rotate(mask, 45))
            train_aug_depths.append(depth)

            train_aug_images.append(rotate(image, -60))
            train_aug_masks.append(rotate(mask, -60))
            train_aug_depths.append(depth)
            
            train_aug_images.append(rotate(image, 30))
            train_aug_masks.append(rotate(mask, 30))
            train_aug_depths.append(depth)
            
            for crop in crops:
                crop_img, crop_mask = crop(image, mask)
                train_aug_images.append(crop_img)
                train_aug_masks.append(crop_mask)
                train_aug_depths.append(depth)

        print('folder {:d}, aug train images to {:d}'.format(i + 1, len(train_aug_images)))
        train_aug_images = np.array(train_aug_images)
        train_aug_masks = np.array(train_aug_masks)
        train_aug_depths = np.array(train_aug_depths)

        np.save(os.path.join(npy_root, 'train_images_folder_{:d}.npy'.format(i + 1)), train_aug_images)
        np.save(os.path.join(npy_root, 'train_masks_folder_{:d}.npy'.format(i + 1)), train_aug_masks)
        np.save(os.path.join(npy_root, 'train_depths_folder_{:d}.npy'.format(i + 1)), train_aug_depths)


def train_masks_statistic():
    masks = np.load(os.path.join(npy_root, 'raw_masks.npy'))

    num = len(masks)
    average_size = 0
    max_size = 0
    min_size = 101 * 101
    empty_num = 0
    small_masks_num = 0
    sizes = []

    for mask in masks:
        assert mask.shape[0] == 101 and mask.shape[1] == 101

        size = np.count_nonzero(mask)
        if size == 0:
            empty_num += 1
            continue

        average_size += size
        if size > max_size:
            max_size = size

        if size < min_size:
            min_size = size

        if size < 50:
            small_masks_num += 1

        sizes.append(size)

    non_empty_num = num - empty_num
    print('total samples {:d}.'.format(num))
    print('empty {:d}.'.format(empty_num))
    print('max mask size {:d}.'.format(max_size))
    print('min mask size {:d}.'.format(min_size))
    print('average mask size {:f}.'.format(average_size / non_empty_num))
    print('small mask num {:d}.'.format(small_masks_num))
    plt.hist(sizes, bins='auto')
    plt.title('mask size histogram.')
    plt.show()


def load_depth_values():
    depth_csv_path = os.path.join(raw_data_root, 'depths.csv')
    depth_csv = pd.read_csv(depth_csv_path)

    ids = []
    depths = []

    for _, row in depth_csv.iterrows():
        image_id = row['id']
        depth = row['z']
        ids.append(image_id)
        depths.append(depth)

    max_d = max(depths)
    min_d = min(depths)
    depth_dict = {}

    for id, depth in zip(ids, depths):
        normed_d = (depth - min_d) / (max_d - min_d)
        depth_dict[id] = normed_d

    return depth_dict


if __name__ == '__main__':
    prepare_train_data()
    prepare_test_data()
    create_n_folder(6)
