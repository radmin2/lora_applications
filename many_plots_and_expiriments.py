import os
import time
import json
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
from sklearn.decomposition import PCA
from datasets import DatasetDict

# !!! ИМПОРТ МЕТРИК !!!
try:
    from metrics_utils import ModelEvaluator
except ImportError:
    raise ImportError("Файл 'metrics_utils.py' должен быть в той же папке!")

# ==========================================
# 1. НАСТРОЙКИ И ПУТИ
# ==========================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join("experiment_results", timestamp)
os.makedirs(RESULTS_DIR, exist_ok=True)

CACHE_DIR = "local_cache"
DATA_CACHE_PATH = os.path.join(CACHE_DIR, "tokenized_imdb")
MODEL_CACHE_PATH = os.path.join(CACHE_DIR, "distilbert_base")

print(f"🚀 Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"📂 Results: {os.path.abspath(RESULTS_DIR)}")
print(f"💾 Cache: {os.path.abspath(CACHE_DIR)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_plot(fig, name, subfolder=""):
    path = os.path.join(RESULTS_DIR, subfolder)
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, name)
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {file_path}")

# ==========================================
# 2. ПОДГОТОВКА ДАННЫХ (С КЭШИРОВАНИЕМ)
# ==========================================
def get_data_and_tokenizer():
    if not os.path.exists(MODEL_CACHE_PATH):
        print("📥 Caching model/tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model_base = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
        tokenizer.save_pretrained(MODEL_CACHE_PATH)
        model_base.save_pretrained(MODEL_CACHE_PATH)
    else:
        print("📦 Loading tokenizer from cache...")
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_CACHE_PATH)

    if os.path.exists(DATA_CACHE_PATH):
        print("📦 Loading tokenized dataset from cache...")
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

        tokenized_datasets = DatasetDict({"train": tokenized_train, "test": tokenized_test})
        tokenized_datasets.save_to_disk(DATA_CACHE_PATH)

        train_data = tokenized_train
        test_data = tokenized_test

    return train_data, test_data, tokenizer

train_dataset, test_dataset, tokenizer = get_data_and_tokenizer()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, num_workers=2, pin_memory=True)

# ==========================================
# 3. LoRA LAYER (✅ ИСПРАВЛЕНО)
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
# 4. ИНСТРУМЕНТЫ АНАЛИЗА
# ==========================================
def get_activations(model, loader, limit_batches=10):
    """Сбор активаций CLS токена для PCA"""
    model.eval()
    activations = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= limit_batches: break
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            outputs = model.distilbert(input_ids, attention_mask=mask)
            cls_acts = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            activations.append(cls_acts)
    return np.vstack(activations)

def analyze_lora_layer(model):
    """SVD и Heatmap первого LoRA слоя"""
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            W_delta = (module.lora_a @ module.lora_b).detach().cpu().numpy() * module.scaling
            
            try:
                _, S, _ = np.linalg.svd(W_delta, full_matrices=False)
            except: 
                S = np.zeros(10)
            
            return S, W_delta
    return None, None

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
# 5. УНИВЕРСАЛЬНЫЙ ЦИКЛ ОБУЧЕНИЯ
# ==========================================
def run_experiment(exp_name, mode="lora", learning_rate=1e-4, epochs=10, lora_rank=8):
    print(f"\n>>> 🧪 {exp_name} (Mode={mode}, LR={learning_rate})")
    
    # 1. Загрузка модели
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_CACHE_PATH, num_labels=2).to(device)

    # 2. Настройка режима
    if mode == "lora":
        model = apply_lora(model, rank=lora_rank)
        # ✅ ЗАМОРОЗИМ классификатор
        # Раскомментируйте, если нужно обучать голову:
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
    
    # 3. Метрики "ДО"
    print("   📸 Capturing 'Before' state...")
    acts_before = get_activations(model, test_loader)
    
    # 4. Оптимизатор и Шедулер
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1*total_steps), 
        num_training_steps=total_steps
    )
    
    history = {
        'time': [], 'loss': [], 'step': [], 'lr': [], 'grad_norm': [], 
        'lora_b_norm': []
    }
    
    start_time = time.time()
    model.train()
    global_step = 0
    
    print("   🔥 Training...")
    for epoch in range(epochs):
        for batch in tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            # Grad Norm (до клиппинга)
            total_norm = 0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            history['grad_norm'].append(total_norm)
            
            # LoRA B Norm
            if mode == "lora":
                for m in model.modules():
                    if isinstance(m, LoRALayer):
                        history['lora_b_norm'].append(m.lora_b.data.norm(2).item())
                        break
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            history['loss'].append(loss.item())
            history['lr'].append(scheduler.get_last_lr()[0])
            history['time'].append(time.time() - start_time)
            history['step'].append(global_step)
            global_step += 1

    # 5. Метрики "ПОСЛЕ"
    print("   📸 Capturing 'After' state...")
    acts_after = get_activations(model, test_loader)
    
    analysis_data = {
        'acts_before': acts_before,
        'acts_after': acts_after,
        'svd_s': None,
        'w_delta_heatmap': None
    }
    
    if mode == "lora":
        S, W_delta = analyze_lora_layer(model)
        analysis_data['svd_s'] = S
        analysis_data['w_delta_heatmap'] = W_delta

    # 6. Генерация персональных графиков
    plot_experiment_details(exp_name, history, analysis_data, mode)

    return history, model

# ==========================================
# 6. ОТРИСОВКА ГРАФИКОВ
# ==========================================
def plot_experiment_details(exp_name, history, analysis, mode):
    """Дашборд для одного эксперимента"""
    sns.set(style="whitegrid")
    
    # 1. Основные метрики (Loss, LR, Grad)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    def smooth(scalars, weight=0.9):
        if len(scalars) == 0: return []
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    # Loss
    axes[0].plot(smooth(history['loss']), label='Train Loss', color='blue', alpha=0.7)
    axes[0].set_title(f"{exp_name}: Loss")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss (Smoothed)")
    axes[0].grid(alpha=0.3)
    
    # LR
    axes[1].plot(history['lr'], color='purple', lw=1.5)
    axes[1].set_title("Learning Rate Schedule")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("LR")
    axes[1].grid(alpha=0.3)
    
    # Grad Norm
    axes[2].plot(smooth(history['grad_norm']), color='orange', alpha=0.6, lw=1)
    axes[2].set_title("Gradient Norm (Pre-clip)")
    axes[2].set_xlabel("Steps")
    axes[2].grid(alpha=0.3)
    
    save_plot(fig, f"{exp_name}_metrics.png")
    
    # 2. LoRA Specifics
    if mode == "lora" and len(history['lora_b_norm']) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # LoRA B Norm
        axes[0].plot(history['lora_b_norm'], color='green', lw=1.5)
        axes[0].set_title("LoRA Matrix B Norm (Growth)")
        axes[0].set_xlabel("Steps")
        axes[0].grid(alpha=0.3)
        
        # SVD
        if analysis['svd_s'] is not None and len(analysis['svd_s']) > 0:
            axes[1].plot(analysis['svd_s'][:min(30, len(analysis['svd_s']))], marker='o', markersize=4)
            axes[1].set_yscale('log')
            axes[1].set_title("SVD Spectrum of ΔW")
            axes[1].set_xlabel("Singular Value Index")
            axes[1].grid(alpha=0.3)
            
        # Heatmap
        if analysis['w_delta_heatmap'] is not None:
            sns.heatmap(analysis['w_delta_heatmap'][:50, :50], ax=axes[2], cmap="RdBu_r", center=0, cbar=True)
            axes[2].set_title("Heatmap ΔW (50×50)")
            
        save_plot(fig, f"{exp_name}_lora_analysis.png")

    # 3. PCA Drift
    if analysis['acts_before'] is not None and analysis['acts_after'] is not None:
        try:
            pca = PCA(n_components=2)
            combined = np.vstack([analysis['acts_before'], analysis['acts_after']])
            reduced = pca.fit_transform(combined)
            n = len(analysis['acts_before'])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(reduced[:n, 0], reduced[:n, 1], label='Before', alpha=0.4, s=10, c='blue')
            ax.scatter(reduced[n:, 0], reduced[n:, 1], label='After', alpha=0.4, s=10, c='red')
            ax.set_title(f"{exp_name}: Activation Drift (PCA)")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.legend()
            ax.grid(alpha=0.3)
            save_plot(fig, f"{exp_name}_pca_drift.png")
        except Exception as e:
            print(f"⚠️ PCA failed: {e}")

def plot_comparison(all_results):
    """Сравнительный график"""
    plt.figure(figsize=(14, 7))
    
    def smooth(scalars, weight=0.9):
        if len(scalars) == 0: return []
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    for name, hist in all_results.items():
        plt.plot(hist['time'], smooth(hist['loss']), label=name, lw=2, alpha=0.8)
        
    plt.xlabel("Time (seconds)")
    plt.ylabel("Train Loss (Smoothed)")
    plt.title("Performance Comparison: Loss vs Wall-Clock Time")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    save_plot(plt.gcf(), "FINAL_COMPARISON.png")

# ==========================================
# 7. ЗАПУСК ЭКСПЕРИМЕНТОВ
# ==========================================
if __name__ == "__main__":
    all_results = {}
    all_models = {}
    
    # ✅ Создаём evaluator
    evaluator = ModelEvaluator(save_dir=os.path.join(RESULTS_DIR, "metrics"))
    
    EPOCHS = 8  # Можно 10-15 для чёткой разницы

    # --- 1. BASELINE LoRA ---
    hist, model = run_experiment("LoRA_Base", mode="lora", learning_rate=1e-4, epochs=EPOCHS)
    all_results['LoRA (LR 1e-4)'] = hist
    all_models['LoRA_Base'] = model
    
    # --- 2. LoRA High/Low LR ---
    hist, model = run_experiment("LoRA_HighLR", mode="lora", learning_rate=5e-4, epochs=EPOCHS)
    all_results['LoRA (LR 5e-4)'] = hist
    all_models['LoRA_HighLR'] = model
    
    hist, model = run_experiment("LoRA_LowLR", mode="lora", learning_rate=5e-5, epochs=EPOCHS)
    all_results['LoRA (LR 5e-5)'] = hist
    all_models['LoRA_LowLR'] = model
    
    # --- 3. Fine-Tuning ---
    hist, model = run_experiment("FT_1Layer", mode="ft_last_layer", learning_rate=1e-4, epochs=EPOCHS)
    all_results['FT 1-Layer'] = hist
    all_models['FT_1Layer'] = model
    
    hist, model = run_experiment("FT_2Layer_High", mode="ft_last_2_layers", learning_rate=1e-4, epochs=EPOCHS)
    all_results['FT 2-Layer (1e-4)'] = hist
    all_models['FT_2Layer_High'] = model
    
    hist, model = run_experiment("FT_2Layer_Opt", mode="ft_last_2_layers", learning_rate=2e-5, epochs=EPOCHS)
    all_results['FT 2-Layer (2e-5)'] = hist
    all_models['FT_2Layer_Opt'] = model

    # --- 4. Финальное сравнение ---
    print("\n📊 Generating Comparison...")
    plot_comparison(all_results)
    
    # ✅ Собираем метрики
    print("\n🏆 Evaluating all models...")
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