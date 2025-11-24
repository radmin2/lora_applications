import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np

# ==========================================
# 1. НАСТРОЙКИ И ПУТИ
# ==========================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
local_save_dir = os.path.join("experiment_results", timestamp)
os.makedirs(local_save_dir, exist_ok=True)

print(f"Saving results to: {os.path.abspath(local_save_dir)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def save_plot(fig, filename):
    save_path = os.path.join(local_save_dir, filename)
    fig.savefig(save_path)
    print(f"Saved: {filename}")

# ==========================================
# 2. ДАННЫЕ И КЛАСС LoRA
# ==========================================
print("Loading dataset...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
dataset = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

# Берем 2000 примеров для обучения
train_dataset = dataset['train'].shuffle(seed=42).select(range(2000))
train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

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
# 3. ФУНКЦИЯ ОБУЧЕНИЯ
# ==========================================
def run_experiment(mode="lora", learning_rate=1e-4, epochs=10, lora_rank=8):
    print(f"\n>>> Starting Experiment: Mode={mode}, LR={learning_rate}")
    
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    ).to(device)

    if mode == "lora":
        model = apply_lora(model, rank=lora_rank)
        for param in model.classifier.parameters(): param.requires_grad = True
        for param in model.pre_classifier.parameters(): param.requires_grad = True
    
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
    
    history = {'time': [], 'loss': [], 'step': []}
    
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
            
            current_time = time.time() - start_time
            history['time'].append(current_time)
            history['loss'].append(loss.item())
            history['step'].append(global_step)
            global_step += 1
            
    return history

# ==========================================
# 4. ЗАПУСК И ВИЗУАЛИЗАЦИЯ
# ==========================================

# Запускаем сравнение
results = {}
fixed_lr = 1e-4

print("--- Running: LoRA ---")
results['lora_base'] = run_experiment(mode="lora", learning_rate=fixed_lr)

print("--- Running: FT (Last Layer) ---")
results['ft_1_layer'] = run_experiment(mode="ft_last_layer", learning_rate=fixed_lr)

print("--- Running: FT (Last 2 Layers) ---")
results['ft_2_layers'] = run_experiment(mode="ft_last_2_layers", learning_rate=fixed_lr)

# Рисуем
sns.set(style="whitegrid")
def smooth(scalars, weight=0.85):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(results['lora_base']['time'], smooth(results['lora_base']['loss']), label='LoRA')
ax1.plot(results['ft_1_layer']['time'], smooth(results['ft_1_layer']['loss']), label='FT Last Layer', linestyle='--')
ax1.plot(results['ft_2_layers']['time'], smooth(results['ft_2_layers']['loss']), label='FT Last 2 Layers', linestyle='-.')

ax1.set_title("Training Loss vs. Time")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Loss")
ax1.legend()

save_plot(fig1, "loss_vs_time_local.png")
plt.show()

print("Done!")
