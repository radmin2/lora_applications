import os
import time
import math
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm
import numpy as np
from datasets import DatasetDict

# !!! ИМПОРТ МЕТРИК !!!
try:
    from metrics_utils import ModelEvaluator
except ImportError:
    raise ImportError("Файл 'metrics_utils.py' должен быть в той же папке!")

# ==========================================
# 1. НАСТРОЙКИ
# ==========================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
local_save_dir = os.path.join("experiment_results", timestamp)
os.makedirs(local_save_dir, exist_ok=True)

CACHE_DIR = "local_cache"
DATA_CACHE_PATH = os.path.join(CACHE_DIR, "tokenized_imdb")
MODEL_CACHE_PATH = os.path.join(CACHE_DIR, "distilbert_base")

print(f"📂 Results: {os.path.abspath(local_save_dir)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def save_plot(fig, filename):
    save_path = os.path.join(local_save_dir, filename)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {filename}")

def evaluate_model(model, dataloader):
    """Получение предсказаний для метрик"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=mask)
            probs = F.softmax(outputs.logits, dim=1)[:, 1]
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_labels, all_preds

# ==========================================
# 2. ДАННЫЕ
# ==========================================
def get_data():
    if not os.path.exists(MODEL_CACHE_PATH):
        print("📥 Caching model...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
        tokenizer.save_pretrained(MODEL_CACHE_PATH)
        model.save_pretrained(MODEL_CACHE_PATH)
    
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_CACHE_PATH)
    
    if not os.path.exists(DATA_CACHE_PATH):
        print("📥 Processing dataset...")
        dataset = load_dataset("imdb")
        train_ds = dataset['train'].shuffle(seed=42).select(range(5000))
        test_ds = dataset['test'].shuffle(seed=42).select(range(500))

        def tokenize(ex):
            return tokenizer(ex['text'], padding='max_length', truncation=True, max_length=128)

        tokenized_train = train_ds.map(tokenize, batched=True)
        tokenized_test = test_ds.map(tokenize, batched=True)
        tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        
        DatasetDict({"train": tokenized_train, "test": tokenized_test}).save_to_disk(DATA_CACHE_PATH)
        return tokenized_train, tokenized_test
    else:
        print("📦 Loading dataset...")
        ds = load_from_disk(DATA_CACHE_PATH)
        return ds["train"], ds["test"]

train_dataset, test_dataset = get_data()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=2, pin_memory=True)

# ==========================================
# 3. LoRA CLASS (✅ ИСПРАВЛЕНО)
# ==========================================
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.original_layer = original_layer
        
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # ✅ Kaiming инициализация
        self.lora_a = nn.Parameter(torch.empty(original_layer.in_features, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, original_layer.out_features))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.original_layer(x) + ((x @ self.lora_a) @ self.lora_b) * self.scaling

def apply_lora(model, rank=8, alpha=16):
    # ✅ Больше слоёв
    target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]
    for name, module in model.named_modules():
        if name.split('.')[-1] in target_modules and isinstance(module, nn.Linear):
            parent = model.get_submodule(".".join(name.split('.')[:-1]))
            child_name = name.split('.')[-1]
            setattr(parent, child_name, LoRALayer(module, rank, alpha))
    return model

# ==========================================
# 4. ФУНКЦИЯ ОБУЧЕНИЯ
# ==========================================
def run_experiment(mode="lora", learning_rate=1e-4, epochs=10, lora_rank=8, use_scheduler=True):
    """
    mode: lora, ft_last_layer, ft_last_2_layers
    use_scheduler: True - динамический LR, False - константный
    """
    print(f"\n>>> Starting: {mode} | LR={learning_rate} | Scheduler={use_scheduler}")
    
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_CACHE_PATH, num_labels=2
    ).to(device)

    if mode == "lora":
        model = apply_lora(model, rank=lora_rank)
        # ✅ ЗАМОРОЗИМ классификатор для чистоты эксперимента
        # Раскомментируйте, если хотите обучать голову:
        # for param in model.classifier.parameters(): param.requires_grad = True
        # for param in model.pre_classifier.parameters(): param.requires_grad = True
    
    elif mode == "ft_last_layer":
        for param in model.parameters(): param.requires_grad = False
        for param in model.classifier.parameters(): param.requires_grad = True
        for param in model.pre_classifier.parameters(): param.requires_grad = True
        for param in model.distilbert.transformer.layer[-1].parameters(): param.requires_grad = True
        
    elif mode == "ft_last_2_layers":
        for param in model.parameters(): param.requires_grad = False
        for param in model.classifier.parameters(): param.requires_grad = True
        for param in model.pre_classifier.parameters(): param.requires_grad = True
        for param in model.distilbert.transformer.layer[-1].parameters(): param.requires_grad = True
        for param in model.distilbert.transformer.layer[-2].parameters(): param.requires_grad = True

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # ✅ ОПЦИОНАЛЬНЫЙ SCHEDULER
    if use_scheduler:
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1*total_steps), 
            num_training_steps=total_steps
        )
    else:
        scheduler = None
    
    history = {'time': [], 'loss': [], 'step': [], 'lr': []}
    
    start_time = time.time()
    model.train()
    global_step = 0
    
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = learning_rate
            
            history['time'].append(time.time() - start_time)
            history['loss'].append(loss.item())
            history['step'].append(global_step)
            history['lr'].append(current_lr)
            global_step += 1
            
    return history, model

# ==========================================
# 5. ЗАПУСК И ВИЗУАЛИЗАЦИЯ
# ==========================================
if __name__ == "__main__":
    results = {}
    models = {}
    evaluator = ModelEvaluator(save_dir=os.path.join(local_save_dir, "metrics"))
    
    fixed_lr = 1e-4
    EPOCHS = 10

    # ✅ LoRA с ФИКСИРОВАННЫМ LR (без scheduler)
    print("--- LoRA (Fixed LR) ---")
    results['LoRA_Fixed_LR'], models['LoRA_Fixed_LR'] = run_experiment(
        mode="lora", 
        learning_rate=fixed_lr, 
        epochs=EPOCHS,
        use_scheduler=False  # <-- Константный LR
    )

    # ✅ LoRA с ДИНАМИЧЕСКИМ LR (с scheduler)
    print("--- LoRA (Dynamic LR) ---")
    results['LoRA_Dynamic_LR'], models['LoRA_Dynamic_LR'] = run_experiment(
        mode="lora", 
        learning_rate=fixed_lr, 
        epochs=EPOCHS,
        use_scheduler=True  # <-- Warmup + Linear Decay
    )

    # Fine-Tuning
    print("--- FT (Last Layer) ---")
    results['FT_1_Layer'], models['FT_1_Layer'] = run_experiment(
        mode="ft_last_layer", 
        learning_rate=fixed_lr, 
        epochs=EPOCHS
    )

    print("--- FT (Last 2 Layers) ---")
    results['FT_2_Layers'], models['FT_2_Layers'] = run_experiment(
        mode="ft_last_2_layers", 
        learning_rate=fixed_lr, 
        epochs=EPOCHS
    )

    # ✅ Собираем метрики
    for name, model in models.items():
        print(f"📊 Evaluating {name}...")
        y_true, y_prob = evaluate_model(model, test_loader)
        evaluator.add_predictions(name, y_true, y_prob)

    # --- ГРАФИКИ ---
    sns.set(style="whitegrid")
    def smooth(scalars, weight=0.85):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    # График 1: Loss vs Time
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for name, hist in results.items():
        ax1.plot(hist['time'], smooth(hist['loss']), label=name, lw=2)

    ax1.set_title("Training Loss vs. Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Loss (Smoothed)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    save_plot(fig1, "loss_vs_time.png")

    # График 2: Learning Rate Schedule
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for name, hist in results.items():
        ax2.plot(hist['step'], hist['lr'], label=name, lw=1.5, alpha=0.8)
    
    ax2.set_title("Learning Rate Schedule Comparison")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Learning Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    save_plot(fig2, "lr_schedule.png")

    # ✅ Генерация метрик
    print("\n🏆 Generating Metric Reports...")
    evaluator.save_metrics_to_json()
    evaluator.plot_roc_curves()
    evaluator.plot_pr_curves()
    evaluator.plot_confusion_matrices()
    evaluator.plot_metric_bar_chart()

    print(f"\n✅ Done! Check: {local_save_dir}")