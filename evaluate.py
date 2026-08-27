"""在测试集上评估：accuracy / sensitivity / specificity / AUC + ROC 与分割可视化。"""
import os
# 解决 OpenCV(numpy/MKL) 与 torch 的 OpenMP 运行时重复加载冲突，必须在导入任何库前设置
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
import data
from model import UNet


def roc_curve(y_true, y_score):
    """按得分降序累计，返回 (fpr, tpr)。"""
    order = np.argsort(-y_score, kind='mergesort')
    yt = y_true[order].astype(np.float64)
    tps = np.cumsum(yt)
    fps = np.cumsum(1.0 - yt)
    n_pos = yt.sum()
    n_neg = (1.0 - yt).sum()
    tpr = tps / n_pos if n_pos > 0 else np.zeros_like(tps)
    fpr = fps / n_neg if n_neg > 0 else np.zeros_like(fps)
    return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr])


def roc_auc(y_true, y_score):
    fpr, tpr = roc_curve(y_true, y_score)
    return float(np.trapz(tpr, fpr))


def compute_metrics(y_true, y_pred, y_score):
    """在 FOV 内计算各指标。输入均为一维数组。"""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = float(np.logical_and(y_pred, y_true).sum())
    tn = float(np.logical_and(~y_pred, ~y_true).sum())
    fp = float(np.logical_and(y_pred, ~y_true).sum())
    fn = float(np.logical_and(~y_pred, y_true).sum())

    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn) if (tp + fn) else 0.0      # 敏感度 = 召回率
    spe = tn / (tn + fp) if (tn + fp) else 0.0      # 特异度
    pre = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * pre * sen / (pre + sen) if (pre + sen) else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    auc = roc_auc(y_true.astype(np.float64), y_score.astype(np.float64))
    return dict(acc=acc, sens=sen, spec=spe, prec=pre, f1=f1, dice=dice, auc=auc)


def predict_full(model, img, device):
    im, H, W = data.pad_to_multiple(img, 16)
    x = torch.from_numpy(im[None, None].astype(np.float32)).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return score[:H, :W]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(base=config.BASE_FILTERS).to(device)
    ckpt = os.path.join(config.MODEL_DIR, 'unet_drive_best.pt')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print('loaded', ckpt, '| device', device)

    rows = []
    all_score, all_true = [], []
    for iid in config.TEST_IDS:
        img = data.load_test_image(iid)
        lab = data.load_label(iid, 'test')
        msk = data.load_mask(iid, 'test')
        score = predict_full(model, img, device)
        pred = score > config.THRESHOLD
        fov = msk > 0
        m = compute_metrics(lab[fov], pred[fov], score[fov])
        m['id'] = iid
        rows.append(m)
        all_true.append(lab[fov].astype(np.float64))
        all_score.append(score[fov].astype(np.float64))
        print(f"{iid}: acc={m['acc']:.4f} sens={m['sens']:.4f} "
              f"spec={m['spec']:.4f} auc={m['auc']:.4f} dice={m['dice']:.4f}")

    # 保存逐图指标
    keys = ['id', 'acc', 'sens', 'spec', 'prec', 'f1', 'dice', 'auc']
    with open(os.path.join(config.RESULT_DIR, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 平均指标
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys if k != 'id'}
    print('\n=== 测试集平均指标（20 张）===')
    for k, v in means.items():
        print(f'{k}: {v:.4f}')

    # 全局 ROC 曲线
    all_true = np.concatenate(all_true)
    all_score = np.concatenate(all_score)
    fpr, tpr = roc_curve(all_true, all_score)
    auc = roc_auc(all_true, all_score)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC (all test pixels in FOV)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULT_DIR, 'roc_curve.png'), dpi=150)

    # 分割结果对比图（前 6 张测试图）
    show_ids = config.TEST_IDS[:6]
    fig, axes = plt.subplots(3, len(show_ids), figsize=(2.5 * len(show_ids), 7.5))
    for j, iid in enumerate(show_ids):
        img = data.load_test_image(iid)
        lab = data.load_label(iid, 'test')
        pred = predict_full(model, img, device) > config.THRESHOLD
        axes[0, j].imshow(img, cmap='gray'); axes[0, j].set_title(f'{iid} input'); axes[0, j].axis('off')
        axes[1, j].imshow(lab, cmap='gray'); axes[1, j].set_title('ground truth'); axes[1, j].axis('off')
        axes[2, j].imshow(pred, cmap='gray'); axes[2, j].set_title('prediction'); axes[2, j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULT_DIR, 'segmentation.png'), dpi=150)

    print('\n保存完成:')
    print(' -', os.path.join(config.RESULT_DIR, 'metrics.csv'))
    print(' -', os.path.join(config.RESULT_DIR, 'roc_curve.png'))
    print(' -', os.path.join(config.RESULT_DIR, 'segmentation.png'))


if __name__ == '__main__':
    main()
