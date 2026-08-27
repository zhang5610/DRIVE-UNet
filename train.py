"""训练 U-Net 进行视网膜血管分割（GPU/CPU 自适应）。"""
import os
# 解决 OpenCV(numpy/MKL) 与 torch 的 OpenMP 运行时重复加载冲突，必须在导入任何库前设置
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
import data
from model import UNet


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_coef(y_true, y_pred):
    """soft dice，输入均为展平后的浮点张量。"""
    inter = (y_true * y_pred).sum()
    return (2.0 * inter + 1.0) / (y_true.sum() + y_pred.sum() + 1.0)


def bce_dice_loss(logits, y_true):
    """二值交叉熵 + Dice 损失的组合，用于缓解血管前景类别不平衡。"""
    bce = F.binary_cross_entropy_with_logits(logits, y_true)
    pred = torch.sigmoid(logits)
    return bce + (1.0 - dice_coef(y_true, pred))


@torch.no_grad()
def full_image_dice(model, images, labels, device):
    """在整张验证图上计算平均 Dice。images/labels 均为 list of (H,W)。"""
    model.eval()
    dices = []
    for img, lab in zip(images, labels):
        im, H, W = data.pad_to_multiple(img, 16)
        x = torch.from_numpy(im[None, None].astype(np.float32)).to(device)
        out = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        pred = (out[:H, :W] > config.THRESHOLD).astype(np.float32)
        dices.append(dice_coef(torch.from_numpy(lab.astype(np.float32)),
                               torch.from_numpy(pred)).item())
    return float(np.mean(dices))


def main():
    set_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    model = UNet(base=config.BASE_FILTERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'UNet 参数量: {n_params / 1e6:.2f} M')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    sampler = data.PatchSampler(config.TRAIN_IDS)
    val_images = [data.load_train_image(i) for i in config.VAL_IDS]
    val_labels = [data.load_label(i, 'train') for i in config.VAL_IDS]

    history = {'loss': [], 'val_dice': []}
    best_dice = -1.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        for _ in range(config.STEPS_PER_EPOCH):
            X, Y = sampler.sample_batch(config.BATCH_SIZE)
            X = torch.from_numpy(X).to(device)
            Y = torch.from_numpy(Y).to(device)
            optimizer.zero_grad()
            loss = bce_dice_loss(model(X), Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / config.STEPS_PER_EPOCH
        history['loss'].append(avg_loss)

        val_dice = full_image_dice(model, val_images, val_labels, device)
        history['val_dice'].append(val_dice)
        print(f'Epoch {epoch + 1:2d}/{config.EPOCHS}  '
              f'loss={avg_loss:.4f}  val_dice={val_dice:.4f}  '
              f'({time.time() - t0:.1f}s)')

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch + 1
            torch.save(model.state_dict(),
                       os.path.join(config.MODEL_DIR, 'unet_drive_best.pt'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f'早停：验证 dice 连续 {config.PATIENCE} epoch 未提升。')
                break

    torch.save(model.state_dict(),
               os.path.join(config.MODEL_DIR, 'unet_drive_final.pt'))
    print(f'训练完成，best val_dice={best_dice:.4f} @ epoch {best_epoch}')

    # 保存训练历史（画图前先落盘，即使后续异常也不丢）
    np.savez(os.path.join(config.RESULT_DIR, 'history.npz'),
             loss=np.array(history['loss'], dtype=np.float32),
             val_dice=np.array(history['val_dice'], dtype=np.float32))

    # 画 loss / val dice 曲线
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training loss'); plt.xlabel('epoch')
    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'])
    plt.title('Validation Dice'); plt.xlabel('epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULT_DIR, 'loss_curve.png'), dpi=120)
    print('loss curve saved ->', os.path.join(config.RESULT_DIR, 'loss_curve.png'))


if __name__ == '__main__':
    main()
