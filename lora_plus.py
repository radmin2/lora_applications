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
RESULTS_DIR = os.path.join("experiment_lora_plus", timestamp)
os.makedirs(RESULTS_DIR, exist_ok=True)

CACHE_DIR = "local_cache"
DATA_CACHE_PATH = os.path.join(CACHE_DIR, "tokenized_imdb")
MODEL_CACHE_PATH = os.path.join(CACHE_DIR, "distilbert_base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

def save_plot(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path)
    plt.close(fig)

# ==========================================
# 2. ДАННЫЕ (Кэшированные + Быстрые)
# ==========================================
def get_data():
    # Если нет кэша модели - качаем
    if not os.path.exists(MODEL_CACHE_PATH):
        DistilBertTokenizer.from_pretrained('distilbert-base-uncased').save_pretrained(MODEL_CACHE_PATH)
        DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased').save_pretrained(MODEL_CACHE_PATH)

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_CACHE_PATH)

    # Если нет кэша данных - качаем
    if not os.path.exists(DATA_CACHE_PATH):
        print("📥 Processing dataset...")
        dataset = load_dataset("imdb")
        train_ds = dataset['train'].shuffle(seed=42).select(range(2000))
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

# num_workers=2 для скорости!
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
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        self.lora_a = nn.Parameter(torch.randn(original_layer.in_features, rank) * (1 / rank))
        self.lora_b = nn.Parameter(torch.zeros(rank, original_layer.out_features))

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
# 4. ЭКСПЕРИМЕНТ LoRA+ (Разделение LR)
# ==========================================
def run_lora_plus(lora_plus_lambda, base_lr=1e-4, epochs=5):
    exp_name = f"LoRA+ (λ={lora_plus_lambda})"
    print(f"\n>>> 🧪 Testing: {exp_name}")
    
    # 1. Модель
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_CACHE_PATH, num_labels=2).to(device)
    model = apply_lora(model, rank=8, alpha=16)
    for param in model.classifier.parameters(): param.requires_grad = True
    for param in model.pre_classifier.parameters(): param.requires_grad = True
    model.to(device)

    # 2. --- МАГИЯ LoRA+ (Разделение параметров) ---
    params_A_and_others = []
    params_B = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Если это матрица B - ей нужен boost
        if "lora_b" in name:
            params_B.append(param)
        else:
            # lora_a, classifier, pre_classifier идут сюда
            params_A_and_others.append(param)

    # Создаем группы для оптимизатора
    optimizer_grouped_parameters = [
        {
            "params": params_A_and_others, 
            "lr": base_lr, 
            "name": "group_A_base"
        },
        {
            "params": params_B, 
            "lr": base_lr * lora_plus_lambda, # Умножаем LR на коэффициент!
            "name": "group_B_plus"
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    # ----------------------------------------------

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    history = {'loss': [], 'time': []}
    start_time = time.time()
    model.train()
    
    for epoch in range(epochs):
        for batch in tqdm(train_loader, leave=False, desc=f"λ={lora_plus_lambda} Ep {epoch+1}"):
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

    return history

# ==========================================
# 5. ЗАПУСК
# ==========================================
if __name__ == "__main__":
    # Коэффициенты λ для теста
    # 1 = Обычная LoRA
    # 16 = Рекомендуемое значение из статьи LoRA+
    lambdas = [0.125, 0.25, 0.5, 1, 4, 16] 
    
    all_results = {}
    EPOCHS = 5
    BASE_LR = 1e-4

    for lam in lambdas:
        res = run_lora_plus(lora_plus_lambda=lam, base_lr=BASE_LR, epochs=EPOCHS)
        all_results[f"$\lambda$={lam}"] = res

    # Отрисовка
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
        plt.plot(hist['time'], smooth(hist['loss']), label=name, lw=1.5, marker='.', markersize=2, alpha=0.8)

    plt.title(f"LoRA+ Benchmark (Base LR={BASE_LR})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Training Loss (Smoothed)")
    plt.legend(title="Ratio ($\eta_B / \eta_A$)")
    plt.grid(True, alpha=0.3)
    
    save_plot(plt.gcf(), "lora_plus_comparison.png")
    print("\n✅ Done! Results in experiment_lora_plus folder.")