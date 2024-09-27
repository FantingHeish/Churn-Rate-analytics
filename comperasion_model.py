import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加載數據
file_path = '/Users/fangting/Desktop/Code/BankChurners copy.csv'
data = pd.read_csv(file_path)

# 2. 將 Attrition_Flag 轉換為二元數據 (1 = 流失, 0 = 未流失)
data['Attrition_Flag'] = data['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)

# 3. 設置特徵變數和目標變數
features = ['Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 'Income_Category', 
            'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 
            'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 
            'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

X = data[features]

# 4. 處理分類變數進行One-Hot編碼
X = pd.get_dummies(X, drop_first=True)

# 目標變數
y = data['Attrition_Flag']

# 5. 數據標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. 使用PCA進行降維，保留5個主成分
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# 7. 將數據分為訓練集和測試集 (70% 訓練集，30% 測試集)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# 構建與訓練 XGBoost 模型
xgb_model = XGBClassifier(learning_rate=0.2, max_depth=7, n_estimators=200, eval_metric='logloss', use_label_encoder=False)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print(f"XGBoost 準確率: {accuracy_xgb:.2f}")
print("XGBoost 分類報告:")
print(classification_report(y_test, y_pred_xgb))

# 8. 使用邏輯回歸進行訓練和預測
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)

print(f"邏輯回歸準確率: {accuracy_log:.2f}")
print("邏輯回歸分類報告:")
print(classification_report(y_test, y_pred_log))

# 9. 使用隨機森林進行訓練和預測
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"隨機森林準確率: {accuracy_rf:.2f}")
print("隨機森林分類報告:")
print(classification_report(y_test, y_pred_rf))

# 10. 比較模型準確度
model_names = ['XGBoost', 'Logistic Regression', 'Random Forest']
accuracies = [accuracy_xgb, accuracy_log, accuracy_rf]

# 可視化模型準確度比較
plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=accuracies, palette='Blues_d')
plt.title('Comparison of Model Accuracies', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
