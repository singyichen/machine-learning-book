import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# 1. 讀取資料集 raw data
df_text = pd.read_csv("wdbc_missing.csv")

# 分離原始特徵 X 與 字串標籤 y
X = df_text.drop(columns=['target']).to_numpy()
y_raw = df_text['target'].to_numpy()

# 2. 使用 LabelEncoder 將字串標籤 y 轉成 0 和 1 的整數
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

# 查看轉換後的對應關係 (確認依據字母順序排序，B -> 0, M -> 1)
print("\n--- LabelEncoder 轉換對照 ---")
for index, class_name in enumerate(label_encoder.classes_):
    print(f"字串 '{class_name}' 被編碼為整數: {index}")

# 3. 黃金法則：進行任何 Processing 之前，先切分資料集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 4. 建立特徵處理的管線 (Pipeline 只對接特徵矩陣 X)
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), 
    ('scaler', StandardScaler()), 
    ('feature_selection', SelectFromModel(
        LogisticRegression(l1_ratio=1.0, solver='liblinear', random_state=42)
    )),
    ('classifier', LogisticRegression(l1_ratio=0.0, max_iter=2000, random_state=42))
])

# 5. 定義網格 (Search Space) 
# 注意：在 Pipeline 中必須使用「步驟名稱 + 兩個下底線 + 參數名」
param_range = [0.01, 0.1, 1.0, 10.0, 100.0]
param_grid = {
    # 測試特徵選擇 L1 Regularization 強度
    'feature_selection__estimator__C': param_range,
    
    # 測試分類器 L2 Regularization 強度
    'classifier__C': param_range
}

# 6. Model Selection
# 將 scoring 設定為 'recall'
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='recall',
    refit=True,
    n_jobs=-1
)

print("正在進行 5-Fold 交叉驗證與超參數網格搜尋...")
grid_search.fit(X_train, y_train)

print("\n=== Model Selection 最佳化結果 ===")
print(f"最佳超參數組合: {grid_search.best_params_}")
print(f"最佳 Recall 分數: {grid_search.best_score_:.2%}\n")

# 7. Model Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("=== Model Evaluation 最終驗收報告 ===")
print("混淆矩陣 (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))

# 把 0, 1 還原回原來的字串 B, M，印出測試報告
print("\n詳細分類報告 (Classification Report):")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 8. Save Model
# 使用 joblib 進行序列化儲存
import joblib
# 設定儲存的檔案名稱
model_filename = "wdbc_pipeline_lr.pkl"
joblib.dump(best_model, model_filename, compress=3)
print(f"成功將管線（含預處理與模型）儲存至: '{model_filename}'")

# 9. Load Model
loaded_pipeline = joblib.load(model_filename)
print("模型與預處理管線載入成功！")
y_pred = loaded_pipeline.predict(X_test)
print(confusion_matrix(y_test, y_pred))

fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
print("False Negative 的索引位置:", fn_indices)
