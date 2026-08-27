"""DRIVE 数据加载与预处理（纯 numpy/cv2，与深度学习框架无关）。"""
import os
import numpy as np
import cv2
from PIL import Image
import config


def _read_green(path):
    """读取彩色眼底图，取绿色通道（血管对比度最高）。cv2 读入为 BGR。"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img[:, :, 1].astype(np.float32)


def preprocess_image(green):
    """CLAHE 对比度增强 + 归一化到 [0,1]。"""
    green_u8 = np.clip(green, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_u8)
    return enhanced.astype(np.float32) / 255.0


def load_train_image(img_id):
    p = os.path.join(config.TRAIN_DIR, 'images', f'{img_id}_training.tif')
    return preprocess_image(_read_green(p))


def load_test_image(img_id):
    p = os.path.join(config.TEST_DIR, 'images', f'{img_id}_test.tif')
    return preprocess_image(_read_green(p))


def load_label(img_id, split='train'):
    """二值血管标注，形状 (H, W) uint8，取值为 {0, 1}。"""
    if split == 'train':
        p = os.path.join(config.TRAIN_DIR, '1st_manual', f'{img_id}_manual1.gif')
    else:
        p = os.path.join(config.TEST_DIR, '1st_manual', f'{img_id}_manual1.gif')
    lab = np.asarray(Image.open(p))
    return (lab > 0).astype(np.uint8)


def load_mask(img_id, split='train'):
    """FOV 掩膜，形状 (H, W) uint8，取值为 {0, 1}。"""
    if split == 'train':
        p = os.path.join(config.TRAIN_DIR, 'mask', f'{img_id}_training_mask.gif')
    else:
        p = os.path.join(config.TEST_DIR, 'mask', f'{img_id}_test_mask.gif')
    m = np.asarray(Image.open(p))
    return (m > 0).astype(np.uint8)


def pad_to_multiple(img, m=16):
    """反射填充到 m 的整数倍，返回 (填充后图, 原 H, 原 W)。"""
    H, W = img.shape
    ph = (m - H % m) % m
    pw = (m - W % m) % m
    return np.pad(img, ((0, ph), (0, pw)), mode='reflect'), H, W


class PatchSampler:
    """随机采样训练 patch，带数据增强。返回 numpy 数组（NCHW）。"""

    def __init__(self, img_ids, patch_size=config.PATCH_SIZE):
        self.patch_size = patch_size
        self.images, self.labels, self.masks = [], [], []
        for iid in img_ids:
            self.images.append(load_train_image(iid))
            self.labels.append(load_label(iid, 'train'))
            self.masks.append(load_mask(iid, 'train'))
        self.n_images = len(img_ids)

    def _sample_one(self):
        ps = self.patch_size
        for _ in range(50):
            idx = np.random.randint(self.n_images)
            img = self.images[idx]
            lab = self.labels[idx]
            msk = self.masks[idx]
            H, W = img.shape
            x = np.random.randint(0, W - ps + 1)
            y = np.random.randint(0, H - ps + 1)
            p_lab = lab[y:y + ps, x:x + ps]
            p_msk = msk[y:y + ps, x:x + ps]
            if p_msk.mean() < 0.5:                       # 主要落在 FOV 外则跳过
                continue
            if p_lab.mean() < 0.02 and np.random.rand() < 0.7:  # 偏向含血管的 patch
                continue
            break

        p_img = img[y:y + ps, x:x + ps].copy()
        # 数据增强：随机 90° 旋转 + 水平/垂直翻转
        k = np.random.randint(4)
        if k:
            p_img = np.rot90(p_img, k)
            p_lab = np.rot90(p_lab, k)
        if np.random.rand() < 0.5:
            p_img = np.fliplr(p_img)
            p_lab = np.fliplr(p_lab)
        if np.random.rand() < 0.5:
            p_img = np.flipud(p_img)
            p_lab = np.flipud(p_lab)
        return p_img, p_lab

    def sample_batch(self, batch_size):
        ps = self.patch_size
        X = np.zeros((batch_size, 1, ps, ps), np.float32)
        Y = np.zeros((batch_size, 1, ps, ps), np.float32)
        for i in range(batch_size):
            im, lb = self._sample_one()
            X[i, 0] = im
            Y[i, 0] = lb
        return X, Y
