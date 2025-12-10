log_dir = "training_logs"
os.makedirs(log_dir, exist_ok=True)

# Timestamp for log filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(log_dir, f"train_log_{timestamp}.json")

# Storage
history = []
num_epochs =  100
# Initialize this BEFORE the loop
best_f1 = 0.0
best_model_path = "best_model-nih_medvit.pth"
start_epoch = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MedViT_small(num_classes=14, pretrained=False).to(device)

# class_weights = compute_class_weights(df, disease_labels).to(device)
# criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

loss_history, val_loss_history = [], []
train_acc_history, val_acc_history = [], []

early_stop_patience = 5   # stop if no improvement for N epochs
early_stop_counter = 0

if os.path.exists(best_model_path):
    print(f"🔁 Loading checkpoint from {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    best_val_f1 = checkpoint["best_f1"]
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}, best F1: {best_val_f1:.4f}")
