import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# -------------------------------------------------------------------
# 1. 새로 구축된 파생변수 데이터셋 로드
# -------------------------------------------------------------------
X_tr = pd.read_csv('X_tr_unscaled.csv')
y_tr = pd.read_csv('y_tr.csv')['returned']
X_val = pd.read_csv('X_val_unscaled.csv')
y_val = pd.read_csv('y_val.csv')['returned']

print(f"학습 데이터 형태: {X_tr.shape}, 검증 데이터 형태: {X_val.shape}")

# XGBoost를 위한 scale_pos_weight 계산 (정상 대비 반품의 비율 보정)
scale_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

# -------------------------------------------------------------------
# 2. 5가지 모델 정의 (클래스 불균형 고려)
# -------------------------------------------------------------------
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight='balanced', 
        random_state=42, n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05, 
        scale_pos_weight=scale_weight, eval_metric='logloss', 
        early_stopping_rounds=10, random_state=42, n_jobs=-1
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05, 
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'CatBoost': cb.CatBoostClassifier(
        iterations=200, depth=6, learning_rate=0.05, 
        auto_class_weights='Balanced', random_state=42, thread_count=-1,
        verbose=0, early_stopping_rounds=10
    ),
    'HistGradientBoosting': HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, class_weight='balanced', 
        early_stopping=True, random_state=42
    )
}

# -------------------------------------------------------------------
# 3. 모델 학습, 학습 곡선(Loss 추적) 및 성능 평가
# -------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
axes = axes.flatten()

# 기본 임계값 0.5 사용
eval_threshold = 0.5 

for i, (name, model) in enumerate(models.items()):
    print(f"\n=================== [{name}] ===================")
    
    # [학습 진행 및 학습 곡선 데이터 추출]
    if name == 'Random Forest':
        model.fit(X_tr, y_tr)
        axes[i].text(0.5, 0.5, 'Learning Curve not natively\nsupported for RandomForest', 
                     ha='center', va='center', fontsize=12)
        
    elif name == 'XGBoost':
        model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
        results = model.evals_result()
        axes[i].plot(results['validation_0']['logloss'], label='Train Logloss')
        axes[i].plot(results['validation_1']['logloss'], label='Validation Logloss')
        
    elif name == 'LightGBM':
        # LightGBM 콜백을 통한 evaluation 로그 기록
        evals_result = {}
        callbacks = [lgb.early_stopping(10, verbose=False), lgb.record_evaluation(evals_result)]
        model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='logloss', callbacks=callbacks)
        axes[i].plot(evals_result['training']['binary_logloss'], label='Train Logloss')
        axes[i].plot(evals_result['valid_1']['binary_logloss'], label='Validation Logloss')
        
    elif name == 'CatBoost':
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        axes[i].plot(model.evals_result_['learn']['Logloss'], label='Train Logloss')
        axes[i].plot(model.evals_result_['validation']['Logloss'], label='Validation Logloss')
        
    elif name == 'HistGradientBoosting':
        model.fit(X_tr, y_tr)
        axes[i].plot(model.train_score_, label='Train Score (Accuracy)')
        axes[i].plot(model.validation_score_, label='Validation Score')
        axes[i].set_ylabel('Accuracy (not Logloss)')

    # [그래프 꾸미기]
    if name != 'Random Forest':
        axes[i].set_xlabel('Boosting Iterations')
        if name != 'HistGradientBoosting':
            axes[i].set_ylabel('Logloss')
        axes[i].legend()
    axes[i].set_title(f'{name} Learning Curve')

    # [성능 평가: 확률 예측 후 0.5 기준으로 클래스 판별]
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= eval_threshold).astype(int)
    auc_score = roc_auc_score(y_val, probs)
    
    print(f"ROC-AUC Score: {auc_score:.4f}")
    print(f"--- Classification Report (Threshold {eval_threshold}) ---")
    print(classification_report(y_val, preds))
    
    # 간단한 Confusion Matrix 출력
    cm = confusion_matrix(y_val, preds)
    print("Confusion Matrix:")
    print(f"[[정상(TN): {cm[0][0]:<6} | 오탐(FP): {cm[0][1]:<6}]")
    print(f" [미탐(FN): {cm[1][0]:<6} | 반품(TP): {cm[1][1]:<6}]]")

# 남는 6번째 subplot 빈칸 지우기
fig.delaxes(axes[5])
plt.tight_layout()
plt.show()