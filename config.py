"""全局配置：路径与超参数。"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'DRIVE')
TRAIN_DIR = os.path.join(DATA_DIR, 'training')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

# DRIVE 原始图像尺寸
IMAGE_H, IMAGE_W = 584, 565

# ---- 训练超参数 ----
PATCH_SIZE = 48            # 训练 patch 大小
BATCH_SIZE = 64
STEPS_PER_EPOCH = 200      # 每 epoch 采样的 batch 数（200×64 = 12800 个 patch）
EPOCHS = 80                # 最大训练轮数
BASE_FILTERS = 64          # U-Net 首层卷积核数
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 15              # 验证 dice 连续多少 epoch 未提升则早停

# ---- 数据划分（DRIVE 标准：训练 21-40，测试 01-20）----
TRAIN_IDS = [f'{i:02d}' for i in range(21, 37)]   # 16 张训练
VAL_IDS   = [f'{i:02d}' for i in range(37, 41)]   # 4 张验证
TEST_IDS  = [f'{i:02d}' for i in range(1, 21)]    # 20 张测试

THRESHOLD = 0.5            # 二值化阈值

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
