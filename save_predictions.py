"""把每张测试图的预测血管分割单独保存：二值掩膜 + 叠加可视化 + 真值对照。

产物（results/predictions/ 下）：
  {id}_pred.png     二值预测血管掩膜（血管=255，已用 FOV 掩膜去掉黑边）
  {id}_overlay.png  原始眼底图上叠加红色预测血管
  {id}_gt.png       第一人工标注真值（对照）
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
import cv2
import torch

import config
import data
from model import UNet
from evaluate import predict_full


def load_rgb(img_id):
    p = os.path.join(config.TEST_DIR, 'images', f'{img_id}_test.tif')
    return cv2.imread(p, cv2.IMREAD_COLOR)  # BGR


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(base=config.BASE_FILTERS).to(device)
    ckpt = os.path.join(config.MODEL_DIR, 'unet_drive_best.pt')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print('loaded', ckpt, '| device', device)

    outdir = os.path.join(config.RESULT_DIR, 'predictions')
    os.makedirs(outdir, exist_ok=True)

    for iid in config.TEST_IDS:
        img = data.load_test_image(iid)       # 绿通道（用于预测）
        lab = data.load_label(iid, 'test')    # 真值
        msk = data.load_mask(iid, 'test')     # FOV 掩膜
        rgb = load_rgb(iid)                   # 原始彩色图（叠加用）

        score = predict_full(model, img, device)
        # 阈值二值化 + 去掉 FOV 外的伪血管
        pred = ((score > config.THRESHOLD) & (msk > 0)).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(outdir, f'{iid}_pred.png'), pred)
        cv2.imwrite(os.path.join(outdir, f'{iid}_gt.png'), lab * 255)

        overlay = rgb.copy()
        overlay[pred > 0] = (0, 0, 255)       # BGR 红色
        cv2.imwrite(os.path.join(outdir, f'{iid}_overlay.png'), overlay)

    print(f'saved {len(config.TEST_IDS)} 张测试图的预测 -> {outdir}')


if __name__ == '__main__':
    main()
