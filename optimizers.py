import os
import time
import json
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm
import numpy as np
from datasets import DatasetDict

# ==========================================
# 1. НАСТРОЙКИ И ПУТИ
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
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> Saved plot: {path}")

# ==========================================
# 2. ДАННЫЕ (С УСКОРЕННОЙ ЗАГРУЗКОЙ)
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
        print("📦 Loading tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_CACHE_PATH)

    if os.path.exists(DATA_CACHE_PATH):
        print("📦 Loading dataset...")
        tokenized_datasets = load_from_disk(DATA_CACHE_PATH)
        train_data = tokenized_datasets["train"]
        test_data = tokenized_datasets["test"]
    else:
        print("📥 Processing dataset...")
        dataset = load_dataset("imdb")
        # Берем 2000 train, 500 test
        train_ds = dataset['train'].shuffle(seed=42).select(range(2000))
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

# !!! ВАЖНО: num_workers=2 и pin_memory=True для устранения лагов !!!
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=2, pin_memory=True)

# ==========================================
# 3. LoRA CLASS
# ==========================================
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        self.lora_a = nn.Parameter(torch.randn(in_features, rank) * (1 / rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return self.original_layer(x) + ((x @ self.lora_a) @ self.lora_b) * self.scaling

def apply_lora(model, rank=8, alpha=16):
    target_modules = ["q_lin", "v_lin"]
    for name, module in model.named_modules():
        if name.split('.')[-1] in target_modules and isinstance(module, nn.Linear):
            parent = model.get_submodule(".".join(name.split('.')[:-1]))
            child_name = name.split('.')[-1]
            setattr(parent, child_name, LoRALayer(module, rank, alpha))
    return model

# ==========================================
# 4. ЭКСПЕРИМЕНТ С ОПТИМИЗАТОРАМИ
# ==========================================
def run_optimizer_experiment(optimizer_name, learning_rate, epochs=5):
    exp_name = f"{optimizer_name} (LR {learning_rate})"
    print(f"\n>>> 🧪 Testing: {exp_name}")
    
    # 1. Чистая модель + LoRA
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_CACHE_PATH, num_labels=2).to(device)
    model = apply_lora(model, rank=8, alpha=16)
    for param in model.classifier.parameters(): param.requires_grad = True
    for param in model.pre_classifier.parameters(): param.requires_grad = True
    model.to(device)

    # 2. Выбор оптимизатора
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        # SGD обязательно нужен momentum для трансформеров
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # 3. Scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    history = {'time': [], 'loss': [], 'step': []}
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
            global_step += 1
            
    return history

# ==========================================
# 5. ЗАПУСК И ОТРИСОВКА
# ==========================================
if __name__ == "__main__":
    # Конфигурация: Оптимизатор -> Его оптимальный LR
    configs = [
        {"name": "AdamW",   "lr": 1e-4}, # Стандарт
        {"name": "Adam",    "lr": 1e-4}, # Старик Адам
        {"name": "RMSprop", "lr": 5e-5}, # Часто капризный, берем поменьше
        {"name": "SGD",     "lr": 1e-2}  # SGD нужно сильно пинать (LR выше в 100 раз!)
    ]
    
    all_results = {}
    EPOCHS = 5

    for conf in configs:
        res = run_optimizer_experiment(conf["name"], conf["lr"], epochs=EPOCHS)
        key = f"{conf['name']} (LR {conf['lr']})"
        all_results[key] = res

    # --- ГРАФИК 1: Loss vs Time (С точками, чтобы проверить лаги) ---
    plt.figure(figsize=(12, 8))
    
    def smooth(scalars, weight=0.9):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    for name, hist in all_results.items():
        # marker='.' покажет реальные замеры, если линия прямая между точками - значит лаг
        plt.plot(hist['time'], smooth(hist['loss']), label=name, marker='.', markersize=2, alpha=0.8)

    plt.title("Optimizers Comparison: LoRA Training Speed & Quality")
    plt.xlabel("Wall-Clock Time (seconds)")
    plt.ylabel("Training Loss (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_plot(plt.gcf(), "optimizers_comparison.png")
    print("\n✅ Experiment finished!")