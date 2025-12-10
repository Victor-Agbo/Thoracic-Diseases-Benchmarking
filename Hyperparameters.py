# ---------------------------
# 🔹 Reproducible Setup
# ---------------------------
import random, numpy as np, torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------
# 🔹 Model & Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "MedViT_small"
num_classes = 14
pretrained = False
model = MedViT_small(num_classes=num_classes, pretrained=pretrained).to(device)

# ---------------------------
# 🔹 Training Hyperparameters
# ---------------------------
num_epochs = 100
batch_size = 32
learning_rate = 1e-5
weight_decay = 1e-5
optimizer_type = "AdamW"
scheduler_type = "CosineAnnealingLR"
early_stop_patience = 5
loss_function = "BCEWithLogitsLoss"
class_weights = None  # Optional: set if using weighted loss

# ---------------------------
# 🔹 Data & Augmentation
# ---------------------------
train_augmentations = "RandomFlip, RandomCrop, ColorJitter"
val_augmentations = "Resize, CenterCrop"
shuffle_train = True
shuffle_val = False
num_workers = 4

# ---------------------------
# 🔹 Checkpoint & Logging
# ---------------------------
best_model_path = "best_model-nih_medvit.pth"
log_path = "training_logs/train_log_timestamp.json"
log_metrics = ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]

# ---------------------------
# 🔹 Notes for Reproducibility
# ---------------------------
# - All seeds fixed for Python, NumPy, and PyTorch (CPU & CUDA)
# - Deterministic CuDNN operations
# - Full optimizer & scheduler states saved in checkpoints
