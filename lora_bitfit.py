import os
import time
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
        # Чуть больше данных для заметной разницы
        train_ds = dataset['train'].shuffle(seed=42).select(range(3000)) 
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
# 3. LoRA IMPLEMENTATION
# ==========================================
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.original_layer = original_layer
        # Внимание: при инициализации LoRA мы НЕ замораживаем слой здесь жестко,
        # так как BitFit может потребовать разморозки bias внутри этого слоя.
        # Заморозку будем делать глобально в configure_model.
        
        self.lora_a = nn.Parameter(torch.randn(original_layer.in_features, rank) * (1 / rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, original_layer.out_features))

    def forward(self, x):
        return self.original_layer(x) + ((x @ self.lora_a) @ self.lora_b) * self.scaling

def apply_lora_injection(model, rank=8, alpha=16):
    target_modules = ["q_lin", "v_lin"]
    for name, module in model.named_modules():
        if name.split('.')[-1] in target_modules and isinstance(module, nn.Linear):
            parent = model.get_submodule(".".join(name.split('.')[:-1]))
            child_name = name.split('.')[-1]
            setattr(parent, child_name, LoRALayer(module, rank, alpha))
    return model

# ==========================================
# 4. КОНФИГУРАЦИЯ МОДЕЛИ (LoRA vs BitFit)
# ==========================================
def configure_model(mode):
    """
    mode: 'lora', 'bitfit', 'lora_bitfit'
    """
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_CACHE_PATH, num_labels=2)
    
    # 1. Сначала замораживаем ВСЁ
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Логика для LoRA
    if "lora" in mode:
        model = apply_lora_injection(model, rank=8, alpha=16)
        # Размораживаем только A и B матрицы
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    # 3. Логика для BitFit
    if "bitfit" in mode:
        # Размораживаем ВСЕ параметры, в имени которых есть "bias"
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
    
    # 4. Всегда размораживаем голову классификации (Classifier Head)
    # Иначе модель не сможет выучить новую задачу (IMDb)
    for param in model.classifier.parameters(): param.requires_grad = True
    for param in model.pre_classifier.parameters(): param.requires_grad = True
    
    return model.to(device)

def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

# ==========================================
# 5. ЦИКЛ ОБУЧЕНИЯ
# ==========================================
def run_experiment(mode, epochs=5, lr=1e-4):
    print(f"\n>>> 🧪 Experiment: {mode}")
    model = configure_model(mode)
    
    trainable, total = count_params(model)
    print(f"   📊 Params: {trainable} / {total} ({trainable/total*100:.3f}%)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
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
            
    return history, trainable

# ==========================================
# 6. ЗАПУСК И ВИЗУАЛИЗАЦИЯ
# ==========================================
if __name__ == "__main__":
    modes = ["lora", "bitfit", "lora_bitfit"]
    results = {}
    params_counts = {}
    
    EPOCHS = 5
    LR = 1e-4 # Обычно BitFit требует LR чуть выше (например 5e-4), но начнем с равных условий

    for mode in modes:
        hist, count = run_experiment(mode, epochs=EPOCHS, lr=LR)
        results[mode] = hist
        params_counts[mode] = count

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
        plt.plot(hist['time'], smooth(hist['loss']), label=mode, lw=1.5)
    
    plt.title("Loss vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- ГРАФИК 2: Loss vs Steps (Convergence) ---
    plt.subplot(1, 2, 2)
    for mode, hist in results.items():
        plt.plot(hist['step'], smooth(hist['loss']), label=mode, lw=1.5)
    
    plt.title("Loss vs Steps (Convergence Rate)")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_plot(plt.gcf(), "loss_comparison.png")

    # --- ГРАФИК 3: Trainable Parameters (Bar Chart) ---
    plt.figure(figsize=(8, 6))
    names = list(params_counts.keys())
    values = list(params_counts.values())
    
    bars = plt.bar(names, values, color=['blue', 'orange', 'green'])
    plt.title("Number of Trainable Parameters")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.3)
    
    # Добавляем подписи значений над столбцами
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')
        
    save_plot(plt.gcf(), "params_comparison.png")

    print("\n✅ Done! Results saved.")