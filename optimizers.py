import os
import time
import json
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
RESULTS_DIR = os.path.join("experiment_optimizers", timestamp)
os.makedirs(RESULTS_DIR, exist_ok=True)

CACHE_DIR = "local_cache"
DATA_CACHE_PATH = os.path.join(CACHE_DIR, "tokenized_imdb")
MODEL_CACHE_PATH = os.path.join(CACHE_DIR, "distilbert_base")

print(f"🚀 Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"📂 Results: {os.path.abspath(RESULTS_DIR)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_plot(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {path}")

# ==========================================
# 2. ДАННЫЕ
# ==========================================
def get_data_and_tokenizer():
    if not os.path.exists(MODEL_CACHE_PATH):
        print("📥 Downloading model/tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model_base = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
        tokenizer.save_pretrained(MODEL_CACHE_PATH)
        model_base.save_pretrained(MODEL_CACHE_PATH)
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_CACHE_PATH)

    if os.path.exists(DATA_CACHE_PATH):
        print("📦 Loading dataset...")
        tokenized_datasets = load_from_disk(DATA_CACHE_PATH)
        train_data = tokenized_datasets["train"]
        test_data = tokenized_datasets["test"]
    else:
        print("📥 Processing dataset...")
        dataset = load_dataset("imdb")
        train_ds = dataset['train'].shuffle(seed=42).select(range(5000))
        test_ds = dataset['test'].shuffle(seed=42).select(range(500))

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

        tokenized_train = train_ds.map(tokenize_function, batched=True)
        tokenized_test = test_ds.map(tokenize_function, batched=True)
        
        tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        ds_dict = DatasetDict({"train": tokenized_train, "test": tokenized_test})
        ds_dict.save_to_disk(DATA_CACHE_PATH)
        train_data = tokenized_train
        test_data = tokenized_test

    return train_data, test_data, tokenizer

train_dataset, test_dataset, tokenizer = get_data_and_tokenizer()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=2, pin_memory=True)

def evaluate_model(model, dataloader):
    """Получение предсказаний"""
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
# 4. ЭКСПЕРИМЕНТ С ОПТИМИЗАТОРАМИ
# ==========================================
def run_optimizer_experiment(optimizer_name, learning_rate, epochs=8):
    exp_name = f"{optimizer_name} (LR {learning_rate})"
    print(f"\n>>> 🧪 Testing: {exp_name}")
    
    # Модель + LoRA
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_CACHE_PATH, num_labels=2).to(device)
    model = apply_lora(model, rank=8, alpha=16)
    
    # ✅ ЗАМОРОЗИМ классификатор
    # for param in model.classifier.parameters(): param.requires_grad = True
    # for param in model.pre_classifier.parameters(): param.requires_grad = True
    
    model.to(device)

    # Выбор оптимизатора
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1*total_steps), 
        num_training_steps=total_steps
    )
    
    history = {'time': [], 'loss': [], 'step': [], 'lr': []}
    start_time = time.time()
    model.train()
    global_step = 0
    
    for epoch in range(epochs):
        for batch in tqdm(train_loader, leave=False, desc=f"{exp_name} Ep {epoch+1}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            history['loss'].append(loss.item())
            history['time'].append(time.time() - start_time)
            history['step'].append(global_step)
            history['lr'].append(scheduler.get_last_lr()[0])
            global_step += 1
            
    return history, model

# ==========================================
# 5. ЗАПУСК И ОТРИСОВКА
# ==========================================
if __name__ == "__main__":
    # Конфигурация: Оптимизатор -> Оптимальный LR
    configs = [
        {"name": "AdamW",   "lr": 1e-4},
        {"name": "Adam",    "lr": 1e-4},
        {"name": "RMSprop", "lr": 5e-5},
        {"name": "SGD",     "lr": 1e-2}  # SGD нужен большой LR
    ]
    
    all_results = {}
    all_models = {}
    
    # ✅ Создаём evaluator
    evaluator = ModelEvaluator(save_dir=os.path.join(RESULTS_DIR, "metrics"))
    
    EPOCHS = 8

    for conf in configs:
        hist, model = run_optimizer_experiment(conf["name"], conf["lr"], epochs=EPOCHS)
        key = f"{conf['name']} (LR {conf['lr']})"
        all_results[key] = hist
        all_models[conf['name']] = model

    # --- ГРАФИК 1: Loss vs Time ---
    plt.figure(figsize=(14, 7))
    
    def smooth(scalars, weight=0.9):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    for name, hist in all_results.items():
        plt.plot(hist['time'], smooth(hist['loss']), label=name, lw=2, alpha=0.8)

    plt.title("Optimizers Comparison: LoRA Training")
    plt.xlabel("Wall-Clock Time (seconds)")
    plt.ylabel("Training Loss (Smoothed)")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    save_plot(plt.gcf(), "optimizers_loss_vs_time.png")

    # --- ГРАФИК 2: LR Schedule ---
    plt.figure(figsize=(14, 7))
    for name, hist in all_results.items():
        plt.plot(hist['step'], hist['lr'], label=name, lw=1.5, alpha=0.8)
    
    plt.title("Learning Rate Schedules")
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_plot(plt.gcf(), "optimizers_lr_schedule.png")

    # ✅ Собираем метрики
    print("\n🏆 Evaluating all optimizers...")
    for name, model in all_models.items():
        print(f"   Processing {name}...")
        y_true, y_prob = evaluate_model(model, test_loader)
        evaluator.add_predictions(name, y_true, y_prob)
    
    # ✅ Генерируем отчёты
    evaluator.save_metrics_to_json()
    evaluator.plot_roc_curves()
    evaluator.plot_pr_curves()
    evaluator.plot_confusion_matrices()
    evaluator.plot_metric_bar_chart()
    
    print(f"\n✅ Done! Results: {RESULTS_DIR}")