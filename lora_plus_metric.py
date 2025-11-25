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
import random
from transformers import set_seed

# !!! ИМПОРТ УТИЛИТНОГО ФАЙЛА !!!
try:
    from metrics_utils import ModelEvaluator
except ImportError:
    raise ImportError("Пожалуйста, убедись, что файл 'metrics_utils.py' лежит в этой же папке!")

# ==========================================
# 1. НАСТРОЙКИ
# ==========================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join("experiment_lora_plus_metrics", timestamp)
os.makedirs(RESULTS_DIR, exist_ok=True)

CACHE_DIR = "local_cache"
DATA_CACHE_PATH = os.path.join(CACHE_DIR, "tokenized_imdb")
MODEL_CACHE_PATH = os.path.join(CACHE_DIR, "distilbert_base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

def save_plot(fig, name, subfolder=""):
    """Сохранение графика с поддержкой подпапок"""
    if subfolder:
        path = os.path.join(RESULTS_DIR, subfolder)
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, name)
    else:
        file_path = os.path.join(RESULTS_DIR, name)
    
    fig.savefig(file_path, dpi=150)
    plt.close(fig)

def set_deterministic_seed(seed=42):
    """Фиксирует все возможные источники случайности."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)
    print(f"🔒 Random seed fixed to {seed}")

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
        train_ds = dataset['train'].shuffle(seed=42).select(range(10000))
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
# 3. LoRA IMPLEMENTATION (✅✅✅ ИСПРАВЛЕНО!)
# ==========================================
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.original_layer = original_layer
        
        # Замораживаем оригинальный слой
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # ✅✅✅ ПРАВИЛЬНАЯ ИНИЦИАЛИЗАЦИЯ (как в оригинальной LoRA)
        # A - Kaiming, B - нули
        self.lora_a = nn.Parameter(torch.zeros(original_layer.in_features, rank))
        self.lora_b = nn.Parameter(torch.randn(rank, original_layer.out_features))
        # nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))ssss

    def forward(self, x):
        return self.original_layer(x) + ((x @ self.lora_a) @ self.lora_b) * self.scaling

def apply_lora(model, rank=8, alpha=16):
    target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]
    for name, module in model.named_modules():
        if name.split('.')[-1] in target_modules and isinstance(module, nn.Linear):
            parent = model.get_submodule(".".join(name.split('.')[:-1]))
            child_name = name.split('.')[-1]
            setattr(parent, child_name, LoRALayer(module, rank, alpha))
    return model

# ==========================================
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def evaluate_accuracy(model, loader):
    """Быстрый подсчет Accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def get_predictions_for_metrics(model, loader):
    """Получение вероятностей для Metrics Evaluator"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=mask)
            probs = F.softmax(outputs.logits, dim=1)[:, 1]
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

def smooth(scalars, weight=0.85):
    """Сглаживание для графиков"""
    if len(scalars) == 0: 
        return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# ==========================================
# 5. ЭКСПЕРИМЕНТ LoRA+
# ==========================================
def run_lora_plus(lora_plus_lambda, base_lr=1e-4, epochs=10, eval_steps=5):
    exp_name = f"LoRA_Plus_Lambda_{lora_plus_lambda}"
    print(f"\n>>> 🧪 Testing: {exp_name}")
    
    # 1. Модель
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_CACHE_PATH, num_labels=2).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model = apply_lora(model, rank=8, alpha=16)
    
    # ✅ ВАЖНО: Замораживаем классификатор для чистоты эксперимента
    # Иначе эффект λ будет незаметен!
    
    model.to(device)

    # 2. Группировка параметров (LoRA+)
    params_A_and_others = []
    params_B = []

    for name, param in model.named_parameters():
        if not param.requires_grad: 
            continue
        if "lora_b" in name:
            params_B.append(param)
        else:
            params_A_and_others.append(param)

    # ✅ Проверка, что параметры правильно разделены
    print(f"   📋 Trainable params: {len(params_A_and_others)} in A-group, {len(params_B)} in B-group")
    print(f"   📋 B learning rate: {base_lr * lora_plus_lambda:.2e}")

    optimizer_grouped_parameters = [
        {"params": params_A_and_others, "lr": base_lr},
        {"params": params_B, "lr": base_lr * lora_plus_lambda}
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1*total_steps), 
        num_training_steps=total_steps
    )
    
    history = {
        'loss': [],           
        'step': [],           
        'time': [],           
        'val_accuracy': [],   
        'val_step': [],       
        'epoch_metrics': []   
    }
    
    start_time = time.time()
    model.train()
    global_step = 0
    
    # Папка для графиков по эпохам
    epoch_graphs_dir = os.path.join(RESULTS_DIR, exp_name, "per_epoch")
    os.makedirs(epoch_graphs_dir, exist_ok=True)
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_losses = []  
        epoch_step_indices = []
        
        step_in_epoch = 0
        
        for batch in tqdm(train_loader, leave=False, desc=f"λ={lora_plus_lambda} Ep {epoch+1}/{epochs}"):
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
            
            loss_val = loss.item()
            history['loss'].append(loss_val)
            history['step'].append(global_step)
            history['time'].append(time.time() - start_time)
            
            epoch_losses.append(loss_val)
            epoch_step_indices.append(step_in_epoch)
            
            global_step += 1
            step_in_epoch += 1

            # Проверка внутри эпохи
            if global_step % eval_steps == 0:
                val_acc = evaluate_accuracy(model, test_loader)
                history['val_accuracy'].append(val_acc)
                history['val_step'].append(global_step)
                model.train()
        
        # --- КОНЕЦ ЭПОХИ ---
        epoch_time = time.time() - epoch_start_time
        avg_loss = np.mean(epoch_losses)
        
        print(f"   📊 Epoch {epoch+1}/{epochs} | Time: {epoch_time:.1f}s | Avg Loss: {avg_loss:.4f}")
        
        # ГРАФИК ЛОССА ДЛЯ ЭПОХИ
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(epoch_step_indices, epoch_losses, alpha=0.3, color='blue', lw=1, label='Raw Loss')
        ax.plot(epoch_step_indices, smooth(epoch_losses, weight=0.9), 
                color='blue', lw=2, label='Smoothed Loss')
        ax.axhline(y=avg_loss, color='red', linestyle='--', lw=2, 
                   label=f'Average: {avg_loss:.4f}')
        ax.set_title(f'Loss During Epoch {epoch+1} (λ={lora_plus_lambda})', fontsize=14)
        ax.set_xlabel('Step within Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        epoch_dir = os.path.join(epoch_graphs_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        save_plot(fig, "loss_during_epoch.png", 
                 subfolder=os.path.join(exp_name, "per_epoch", f"epoch_{epoch+1}"))
        
        # МЕТРИКИ
        epoch_evaluator = ModelEvaluator(save_dir=epoch_dir)
        y_true, y_prob = get_predictions_for_metrics(model, test_loader)
        epoch_name = f"Epoch_{epoch+1}"
        epoch_evaluator.add_predictions(epoch_name, y_true, y_prob)
        
        metrics = epoch_evaluator.experiments[epoch_name]['metrics']
        history['epoch_metrics'].append({
            'epoch': epoch + 1,
            'accuracy': metrics['Accuracy'],
            'f1': metrics['F1-Score'],
            'auc': metrics['ROC-AUC'],
            'avg_loss': avg_loss,
            'time': epoch_time
        })
        
        epoch_evaluator.save_metrics_to_json(filename=f"metrics_epoch_{epoch+1}.json")
        epoch_evaluator.plot_confusion_matrices()
        epoch_evaluator.plot_roc_curves()
        epoch_evaluator.plot_pr_curves()
        
        print(f"      ✅ Acc: {metrics['Accuracy']:.4f} | F1: {metrics['F1-Score']:.4f} | AUC: {metrics['ROC-AUC']:.4f}")
        
        model.train()

    return history, model

# ==========================================
# 6. ЗАПУСК
# ==========================================
if __name__ == "__main__":
    set_deterministic_seed(42)

    evaluator = ModelEvaluator(save_dir=os.path.join(RESULTS_DIR, "final_metrics"))
    
    lambdas = [0.0625, 1, 16] 
    all_results = {}
    all_epoch_metrics = {}
    
    EPOCHS = 10
    BASE_LR = 1e-4
    EVAL_EVERY_N_STEPS = 100

    for lam in lambdas:
        hist, trained_model = run_lora_plus(
            lora_plus_lambda=lam, 
            base_lr=BASE_LR, 
            epochs=EPOCHS, 
            eval_steps=EVAL_EVERY_N_STEPS
        )
        name = f"LoRA_Plus_Lambda_{lam}"
        all_results[name] = hist
        all_epoch_metrics[name] = hist['epoch_metrics']
        
        print(f"   📊 Calculating full metrics for {name}...")
        y_true, y_prob = get_predictions_for_metrics(trained_model, test_loader)
        evaluator.add_predictions(name, y_true, y_prob)

    # ==========================================
    # 7. ОБЩИЕ ГРАФИКИ
    # ==========================================

    # ГРАФИК 1: Loss vs Steps
    plt.figure(figsize=(14, 6))
    for name, hist in all_results.items():
        plt.plot(hist['step'], smooth(hist['loss']), label=name, lw=2, alpha=0.8)
    plt.title("Training Loss vs Steps (Iterations)")
    plt.xlabel("Steps (Iterations)")
    plt.ylabel("Loss (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(plt.gcf(), "loss_vs_steps.png")

    # ГРАФИК 2: Loss vs Time
    plt.figure(figsize=(14, 6))
    for name, hist in all_results.items():
        plt.plot(hist['time'], smooth(hist['loss']), label=name, lw=2, alpha=0.8)
    plt.title("Training Loss vs Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Loss (Smoothed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(plt.gcf(), "loss_vs_time.png")

    # ГРАФИК 3: Accuracy vs Steps
    plt.figure(figsize=(14, 6))
    for name, hist in all_results.items():
        plt.plot(hist['val_step'], hist['val_accuracy'], label=name, marker='o', markersize=3, lw=2, alpha=0.8)
    plt.title(f"Validation Accuracy vs Steps (Eval every {EVAL_EVERY_N_STEPS} steps)")
    plt.xlabel("Global Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(plt.gcf(), "accuracy_vs_steps.png")

    # ГРАФИК 4: Loss per Epoch
    plt.figure(figsize=(12, 6))
    for name, epoch_data in all_epoch_metrics.items():
        epochs_list = [d['epoch'] for d in epoch_data]
        avg_losses = [d['avg_loss'] for d in epoch_data]
        plt.plot(epochs_list, avg_losses, marker='o', label=name, lw=2.5, markersize=6)
    plt.title("Average Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(plt.gcf(), "loss_per_epoch.png")

    # ГРАФИК 5: Accuracy per Epoch
    plt.figure(figsize=(12, 6))
    for name, epoch_data in all_epoch_metrics.items():
        epochs_list = [d['epoch'] for d in epoch_data]
        accuracies = [d['accuracy'] for d in epoch_data]
        plt.plot(epochs_list, accuracies, marker='o', label=name, lw=2.5, markersize=6)
    plt.title("Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.5, 1.0])
    save_plot(plt.gcf(), "accuracy_per_epoch.png")

    # ГРАФИК 6: Loss + Accuracy combined
    fig, ax1 = plt.subplots(figsize=(14, 7))
    example_name = list(all_epoch_metrics.keys())[0]
    epoch_data = all_epoch_metrics[example_name]
    
    epochs_list = [d['epoch'] for d in epoch_data]
    avg_losses = [d['avg_loss'] for d in epoch_data]
    accuracies = [d['accuracy'] for d in epoch_data]
    
    color1 = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Average Loss', color=color1, fontsize=12)
    ax1.plot(epochs_list, avg_losses, color=color1, marker='o', lw=2.5, markersize=7)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Validation Accuracy', color=color2, fontsize=12)
    ax2.plot(epochs_list, accuracies, color=color2, marker='s', lw=2.5, markersize=7)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title(f'Loss and Accuracy per Epoch ({example_name})', fontsize=14)
    fig.tight_layout()
    save_plot(fig, "loss_and_accuracy_combined.png")

    # ГРАФИК 7: Метрики (Acc/F1/AUC)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name, epoch_data in all_epoch_metrics.items():
        epochs_list = [d['epoch'] for d in epoch_data]
        accuracies = [d['accuracy'] for d in epoch_data]
        f1_scores = [d['f1'] for d in epoch_data]
        aucs = [d['auc'] for d in epoch_data]
        
        axes[0].plot(epochs_list, accuracies, marker='o', label=name, lw=2)
        axes[1].plot(epochs_list, f1_scores, marker='s', label=name, lw=2)
        axes[2].plot(epochs_list, aucs, marker='^', label=name, lw=2)
    
    axes[0].set_title("Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("F1-Score per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1-Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title("ROC-AUC per Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, "metrics_per_epoch.png")

    # ГРАФИК 8: Training Time
    plt.figure(figsize=(12, 6))
    for name, epoch_data in all_epoch_metrics.items():
        epochs_list = [d['epoch'] for d in epoch_data]
        times = [d['time'] for d in epoch_data]
        plt.plot(epochs_list, times, marker='o', label=name, lw=2)
    plt.title("Training Time per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(plt.gcf(), "time_per_epoch.png")

    # ФИНАЛЬНЫЕ МЕТРИКИ
    print("\n🏆 Generating Final Metric Reports...")
    evaluator.save_metrics_to_json()
    evaluator.plot_roc_curves()
    evaluator.plot_pr_curves()
    evaluator.plot_confusion_matrices()
    evaluator.plot_metric_bar_chart()
    
    print(f"\n✅ Done! Check results in: {RESULTS_DIR}")