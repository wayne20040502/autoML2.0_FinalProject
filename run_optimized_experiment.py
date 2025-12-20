# ==========================================
# 1. 系統設定 (安靜模式)
# ==========================================
import sys
import os
import warnings
import logging
import inspect
from pathlib import Path

logging.disable(logging.CRITICAL)

class AggressiveFilter:
    def __init__(self, original_stream):
        self.original_stream = original_stream
    def write(self, message):
        block_keywords = [">>>> Params", "prediction.classifier", "feature_scaling_candidate", "preprocessor.dimensionality"]
        if any(k in message for k in block_keywords): return
        self.original_stream.write(message)
    def flush(self):
        try: self.original_stream.flush()
        except: pass

# ==========================================
# 2. 載入套件
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, confusion_matrix, roc_auc_score, 
    roc_curve, auc
)

current_dir = os.getcwd()
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path: sys.path.insert(0, src_dir)

try:
    from autoprognosis.studies.classifiers import ClassifierStudy
    from autoprognosis.utils.serialization import save_model_to_file, load_model_from_file
    from autoprognosis.plugins.prediction.classifiers import Classifiers
except ImportError:
    print("❌ 找不到 AutoPrognosis，請確認路徑。"); sys.exit(1)

warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='Microsoft JhengHei')
save_dir = os.getcwd()

# ==========================================
# 3. [核磁共振] 物件深度搜尋引擎
# ==========================================

def get_simple_name(obj):
    """從物件名稱猜演算法"""
    raw = type(obj).__name__.lower()
    if "xgboost" in raw: return "XGBoost"
    if "catboost" in raw: return "CatBoost"
    if "forest" in raw: return "Random Forest"
    if "svm" in raw or "svc" in raw: return "Linear SVM"
    if "logistic" in raw: return "Logistic Regression"
    return "Unknown Model"

def recursive_search(obj, target_keys, found_repo, visited, path=""):
    """
    遞迴掃描物件的所有屬性，尋找目標參數
    """
    # 防止無窮迴圈
    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)
    
    # 1. 檢查自己是否是目標
    # 如果 obj 本身就是一個字典 (例如 hyperparameters={...})
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in target_keys and v is not None:
                # 找到寶藏！存起來
                if k not in found_repo: 
                    found_repo[k] = v
            # 繼續往下鑽
            if hasattr(v, '__dict__') or isinstance(v, (dict, list)):
                recursive_search(v, target_keys, found_repo, visited, path + f".{k}")
        return

    # 2. 檢查物件的屬性 (__dict__)
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            # 命中目標 Key (例如 self.C = 1.0)
            if k in target_keys and v is not None:
                if k not in found_repo:
                    found_repo[k] = v
            
            # 排除私有屬性和系統屬性，優化速度
            if k.startswith("__") or k in ['_logger', 'logger']:
                continue
                
            # 繼續往下鑽 (遞迴)
            if hasattr(v, '__dict__') or isinstance(v, (dict, list)):
                recursive_search(v, target_keys, found_repo, visited, path + f".{k}")

    # 3. 特別處理 List (例如 steps=[...])
    if isinstance(obj, (list, tuple)):
        for item in obj:
             recursive_search(item, target_keys, found_repo, visited, path + "[]")

def deep_extract_params(model_obj):
    """
    啟動深度搜尋，針對單一模型物件
    """
    # 定義我們一定要找到的黃金參數
    target_keys = [
        'C', 'kernel', 'gamma',                      # SVM / LR
        'n_estimators', 'max_depth', 'learning_rate', # Trees
        'min_child_weight', 'reg_alpha', 'reg_lambda', # XGB
        'penalty', 'solver', 'l1_ratio',             # LR
        'n_components'                               # PCA
    ]
    
    found_params = {}
    visited = set()
    
    # 開始地毯式搜索
    recursive_search(model_obj, target_keys, found_params, visited)
    
    # 格式化輸出
    result_list = []
    for k, v in found_params.items():
        val_str = f"{v:.6g}" if isinstance(v, (float, int)) and not isinstance(v, bool) else f"{v}"
        result_list.append(f"{k} = {val_str}")
        
    return sorted(result_list)

def print_model_details(model):
    print("\n" + "="*80 + "\n🔬 模型內部參數深度掃描報告 (Deep Scan)\n" + "="*80)
    print("💡 說明：此程式已暴力掃描記憶體中所有變數，以下是找到的關鍵參數。\n")
    
    if hasattr(model, "models") and hasattr(model, "weights"):
        print("【檢測結果】：集成模型 (Ensemble)")
        
        combined = []
        for m, w in zip(model.models, model.weights):
            # 從物件名稱猜是誰
            real_obj = m.model if hasattr(m, "model") else m
            name = get_simple_name(real_obj)
            
            # 啟動掃描
            params = deep_extract_params(real_obj)
            combined.append({'name': name, 'weight': w, 'params': params})
            
        combined = sorted(combined, key=lambda x: x['weight'], reverse=True)
        ranks = ["🥇 第一名", "🥈 第二名", "🥉 第三名"]
        
        for i in range(min(3, len(combined))):
            comp = combined[i]
            print(f"{ranks[i]}：{comp['name']} (權重: {comp['weight']*100:.2f}%)")
            print("-" * 80)
            
            if comp['params']:
                for line in comp['params']:
                    print(f"   👉 {line}")
            else:
                print("   (⚠️ 掃描完畢但未發現目標參數，可能是使用預設值或名稱被混淆)")
            print("\n")
    else:
        print("【檢測結果】：單一模型 (Single Algorithm)")
        real_obj = model
        name = get_simple_name(real_obj)
        params = deep_extract_params(real_obj)
        
        print(f"🥇 唯一冠軍：{name}")
        print("-" * 80)
        for line in params:
            print(f"   👉 {line}")

# ==========================================
# 4. 繪圖與其他邏輯 (不變)
# ==========================================
def analyze_and_save_chart(y_true, y_pred, y_proba, title):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted'),
        "Recall": recall_score(y_true, y_pred, average='weighted'),
        "F1 Score": f1_score(y_true, y_pred, average='weighted'),
    }
    if y_proba is not None: metrics["AUC-ROC"] = roc_auc_score(y_true, y_proba)

    print(f"\n========================================\n🏆 {title}\n========================================")
    for name, value in metrics.items(): print(f"{name:<15} : {value:.4f}")
    print("-" * 40)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f"Confusion Matrix\n{title}")
    
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color='orange', label=f'AUC = {roc_auc:.4f}')
        axes[1].plot([0, 1], [0, 1], '--')
        axes[1].set_title(f"ROC Curve\n{title}")
        axes[1].legend(loc="lower right")
    
    plt.tight_layout()
    filename = f"chart_{title.replace(' ', '_')}.png"
    plt.savefig(os.path.join(save_dir, filename))
    print(f"📊 圖表已存至: {filename}")
    plt.close()

def perform_dry_run(train_df, target_col, classifiers, feature_selection):
    print("\n🔥 [2/2] 冒煙測試 (Dry Run)...")
    sample_size = min(100, int(len(train_df) * 0.5))
    small_df = train_df.sample(n=sample_size, random_state=42)
    if small_df[target_col].nunique() < 2:
        small_df = pd.concat([small_df, train_df.iloc[:5]]) 

    try:
        test_study = ClassifierStudy(
            study_name="dry_run", dataset=small_df, target=target_col,
            num_iter=1, timeout=60, classifiers=[classifiers[0]], metric="aucroc",
            feature_selection=[feature_selection[0]], score_threshold=0.0 
        )
        if test_study.fit() is None: raise RuntimeError("None Model")
        print("✅ 通過")
    except Exception as e:
        sys.stdout = sys.__stdout__; sys.stderr = sys.__stderr__
        print(f"\n❌ 測試失敗: {e}"); raise e

def main():
    # ==========================================
    # ⚙️ 設定區
    # ==========================================
    FORCE_RETRAIN = False 
    
    print("="*60)
    print(f"   AutoPrognosis 糖尿病預測 (核磁共振版)")
    print("="*60)
    
    file_path = r"C:\Users\aa803\Downloads\archive\diabetes.csv"
    if not os.path.exists(file_path): print(f"❌ 找不到: {file_path}"); return

    df = pd.read_csv(file_path)
    target = df.columns[8]
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df[target])
    
    model_filename = "diabetes_best_model.p"
    model_path = Path(model_filename)
    model = None

    if model_path.exists() and not FORCE_RETRAIN:
        print(f"\n📂 發現模型：{model_filename}")
        print("⏩ 直接載入...")
        model = load_model_from_file(model_path)
        print("✅ 載入成功！")
    else:
        if FORCE_RETRAIN: print(f"\n⚠️  已開啟強制重練...")
        else: print(f"\n🚀 未發現模型，準備訓練...")
            
        classifiers = ["xgboost", "random_forest", "logistic_regression", "catboost", "linear_svm"]
        valid_classifiers = [c for c in classifiers if c in Classifiers().list_available()]
        feature_selection = ["pca", "variance_threshold", "nop"]

        perform_dry_run(train_df, target, valid_classifiers, feature_selection)
        
        sys.stdout = AggressiveFilter(sys.stdout)
        sys.stderr = AggressiveFilter(sys.stderr)
        
        print("\n🚀 [正式開始] AutoML 搜尋 (約 10 分鐘)...")
        TOTAL_TIME_LIMIT = 600
        time_per = int(TOTAL_TIME_LIMIT / len(valid_classifiers))
        
        study = ClassifierStudy(
            study_name="diabetes_run", dataset=train_df, target=target,
            num_iter=20, timeout=time_per, classifiers=valid_classifiers,
            metric="aucroc", feature_selection=feature_selection 
        )
        model = study.fit()
        
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        if model is None: print("❌ 搜尋失敗。"); return
        save_model_to_file(model_path, model)
        print("\n💾 模型已儲存。")

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    y_pred = model.predict(X_test).to_numpy().ravel()
    try: y_proba = model.predict_proba(X_test).to_numpy()[:, 1]
    except: y_proba = None

    analyze_and_save_chart(y_test, y_pred, y_proba, "最佳模型驗證")
    print_model_details(model)

if __name__ == "__main__":
    main()