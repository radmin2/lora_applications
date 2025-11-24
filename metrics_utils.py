import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
)
import torch.nn.functional as F      # Нужно для расчета вероятностей (Softmax)
from metrics_utils import ModelEvaluator # Наш новый класс

class ModelEvaluator:
    def __init__(self, save_dir="evaluation_results"):
        """
        Класс для сбора предсказаний, подсчета метрик и рисования сравнительных графиков.
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # Словарь для хранения данных: { 'Model_Name': {'y_true': [], 'y_prob': [], 'metrics': {...}} }
        self.experiments = {}

    def add_predictions(self, model_name, y_true, y_prob):
        """
        Добавляет результаты модели.
        y_true: истинные лейблы (0 или 1)
        y_prob: вероятности класса 1 (например, после softmax/sigmoid)
        """
        # Приводим к numpy
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = (y_prob > 0.5).astype(int)

        # Считаем скалярные метрики
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, y_prob)
        }

        self.experiments[model_name] = {
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "metrics": metrics
        }
        print(f"✅ [Evaluator] Added {model_name}: F1={metrics['F1-Score']:.4f}, AUC={metrics['ROC-AUC']:.4f}")

    def save_metrics_to_json(self, filename="all_metrics.json"):
        """Сохраняет только численные метрики в JSON для истории."""
        data_to_save = {name: data['metrics'] for name, data in self.experiments.items()}
        path = os.path.join(self.save_dir, filename)
        with open(path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"💾 Metrics saved to {path}")

    def plot_roc_curves(self):
        """Рисует ROC-кривые всех моделей на одном графике."""
        plt.figure(figsize=(10, 8))
        
        for name, data in self.experiments.items():
            fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'])
            auc_val = data['metrics']['ROC-AUC']
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_val:.3f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        self._save_plot("comparison_roc_curve.png")

    def plot_pr_curves(self):
        """Рисует Precision-Recall кривые."""
        plt.figure(figsize=(10, 8))
        
        for name, data in self.experiments.items():
            precision, recall, _ = precision_recall_curve(data['y_true'], data['y_prob'])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f'{name} (AUC = {pr_auc:.3f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve Comparison')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        self._save_plot("comparison_pr_curve.png")

    def plot_confusion_matrices(self):
        """Рисует Confusion Matrix для каждой модели (сеткой)."""
        n = len(self.experiments)
        if n == 0: return

        cols = min(3, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten() if n > 1 else [axes]

        for i, (name, data) in enumerate(self.experiments.items()):
            cm = confusion_matrix(data['y_true'], data['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
            axes[i].set_title(f"{name}\nF1: {data['metrics']['F1-Score']:.3f}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")

        # Удаляем пустые оси, если есть
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        self._save_plot("comparison_confusion_matrices.png")

    def plot_metric_bar_chart(self):
        """Сравнивает основные метрики столбиками."""
        df_list = []
        for name, data in self.experiments.items():
            m = data['metrics']
            df_list.append({
                "Model": name, "Accuracy": m["Accuracy"], "F1": m["F1-Score"], "AUC": m["ROC-AUC"]
            })
        
        df = pd.DataFrame(df_list)
        df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="viridis")
        plt.title("Model Performance Comparison")
        plt.ylim(0.5, 1.0) # Обычно метрики выше 0.5, так виднее разницу
        plt.grid(axis='y', alpha=0.3)
        
        self._save_plot("comparison_bar_chart.png")

    def _save_plot(self, filename):
        path = os.path.join(self.save_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"🖼 Plot saved: {path}")
