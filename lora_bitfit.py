import os
import time
import math
import shutil
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

# !!! ИМПОРТ УТИЛИТНОГО ФАЙЛА !!!
try:
    from metrics_utils import ModelEvaluator
except ImportError:
    raise ImportError("Файл 'metrics_utils.py' должен быть в той же папке!")

# ==========================================
# 1. НАСТРОЙКИ
# ==========================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join("experiment_bitfit", timestamp)
os.makedirs(RESULTS_DIR, exist_ok=True)

CACHE_DIR = "local_cache"
DATA_CACHE_PATH = os.path.join(CACHE_DIR, "tokenized_imdb")
MODEL_CACHE_PATH = os.path.join(CACHE_DIR, "distilbert_base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")
print(f"📂 Results: {os.path.abspath(RESULTS_DIR)}")

def save_plot(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path)
    plt.close(fig)

def evaluate_model(model, dataloader):
    """Получает предсказания для ModelEvaluator"""
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
        DistilBertTokenizer.from_pretrained('distilbert-base-uncased').save_pretrained(MODEL_CACHE_PATH)
        DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased').save_pretrained(MODEL_CACHE_PATH)

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
        print("📦 Loading dataset from cache...")
        ds = load_from_disk(DATA_CACHE_PATH)
        return ds["train"], ds["test"]

train_dataset, test_dataset = get_data()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=2, pin_memory=True)

# ==========================================
# 3. LoRA IMPLEMENTATION (✅ ИСПРАВЛЕНО)
# ==========================================
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.original_layer = original_layer
        
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # ✅ ПРАВИЛЬНАЯ инициализация (Kaiming)
        self.lora_a = nn.Parameter(torch.empty(original_layer.in_features, rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, original_layer.out_features))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.original_layer(x) + ((x @ self.lora_a) @ self.lora_b) * self.scaling

def apply_lora_injection(model, rank=8, alpha=16):
    # ✅ Больше слоёв
    target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]
    for name, module in model.named_modules():
        if name.split('.')[-1] in target_modules and isinstance(module, nn.Linear):
            parent = model.get_submodule(".".join(name.split('.')[:-1]))
            child_name = name.split('.')[-1]
            setattr(parent, child_name, LoRALayer(module, rank, alpha))
    return model

# ==========================================
# 4. КОНФИГУРАЦИЯ МОДЕЛИ
# ==========================================
def configure_model(mode):
    """mode: 'lora', 'bitfit', 'lora_bitfit'"""
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_CACHE_PATH, num_labels=2)
    
    # Замораживаем всё
    for param in model.parameters():
        param.requires_grad = False
        
    # LoRA
    if "lora" in mode:
        model = apply_lora_injection(model, rank=8, alpha=16)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    # BitFit
    if "bitfit" in mode:
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
    
    # ✅ КЛАССИФИКАТОР ЗАМОРОЖЕН (для чистоты эксперимента)
    # Если хотите разморозить - раскомментируйте:
    # for param in model.classifier.parameters(): param.requires_grad = True
    # for param in model.pre_classifier.parameters(): param.requires_grad = True
    
    return model.to(device)

def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

# ==========================================
# 5. ЦИКЛ ОБУЧЕНИЯ
# ==========================================
def run_experiment(mode, epochs=10, lr=1e-4):
    print(f"\n>>> 🧪 Experiment: {mode}")
    model = configure_model(mode)
    
    trainable, total = count_params(model)
    print(f"   📊 Params: {trainable:,} / {total:,} ({trainable/total*100:.3f}%)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1*total_steps), 
        num_training_steps=total_steps
    )
    
    history = {'loss': [], 'time': [], 'step': []}
    start_time = time.time()
    model.train()
    global_step = 0
    
    for epoch in range(epochs):
        for batch in tqdm(train_loader, leave=False, desc=f"{mode} Ep {epoch+1}"):
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
            global_step += 1
            
    return history, model

# ==========================================
# 6. ЗАПУСК И ВИЗУАЛИЗАЦИЯ
# ==========================================
if __name__ == "__main__":
    modes = ["lora", "bitfit", "lora_bitfit"]
    results = {}
    params_counts = {}
    
    # ✅ Создаём evaluator
    evaluator = ModelEvaluator(save_dir=os.path.join(RESULTS_DIR, "metrics"))
    
    EPOCHS = 10
    LR = 5e-4  # BitFit обычно лучше с чуть большим LR

    for mode in modes:
        hist, trained_model = run_experiment(mode, epochs=EPOCHS, lr=LR)
        results[mode] = hist
        params_counts[mode] = count_params(trained_model)[0]
        
        # ✅ Собираем метрики
        print(f"   📊 Evaluating {mode}...")
        y_true, y_prob = evaluate_model(trained_model, test_loader)
        evaluator.add_predictions(mode.upper(), y_true, y_prob)

    # --- ГРАФИК 1: Loss vs Time ---
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    
    def smooth(scalars, weight=0.9):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    for mode, hist in results.items():
        plt.plot(hist['time'], smooth(hist['loss']), label=mode.upper(), lw=2)
    
    plt.title("Loss vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Loss (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- ГРАФИК 2: Loss vs Steps ---
    plt.subplot(1, 2, 2)
    for mode, hist in results.items():
        plt.plot(hist['step'], smooth(hist['loss']), label=mode.upper(), lw=2)
    
    plt.title("Loss vs Steps")
    plt.xlabel("Steps")
    plt.ylabel("Loss (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_plot(plt.gcf(), "loss_comparison.png")

    # --- ГРАФИК 3: Trainable Parameters ---
    plt.figure(figsize=(8, 6))
    names = [m.upper() for m in params_counts.keys()]
    values = list(params_counts.values())
    
    bars = plt.bar(names, values, color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.title("Trainable Parameters Comparison")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{int(yval):,}', 
                 va='bottom', ha='center', fontsize=10)
        
    save_plot(plt.gcf(), "params_comparison.png")

    # ✅ ГЕНЕРИРУЕМ МЕТРИКИ
    print("\n🏆 Generating Metric Reports...")
    evaluator.save_metrics_to_json()
    evaluator.plot_roc_curves()
    evaluator.plot_pr_curves()
    evaluator.plot_confusion_matrices()
    evaluator.plot_metric_bar_chart()

    print(f"\n✅ Done! Results saved to: {RESULTS_DIR}")