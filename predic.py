import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 載入數據
file_path = '/Users/fangting/Desktop/Code/BankChurners copy.csv'  # 替換為你的實際路徑
data = pd.read_csv(file_path)

# 2. 數據處理
# 將 Attrition_Flag 轉換為二元數據 (1 = 流失, 0 = 未流失)
data['Attrition_Flag'] = data['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)

# 選擇需要的特徵變數
features = ['Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Revolving_Bal', 'Credit_Limit',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

# 3. 計算高風險和低風險閾值 (25% 和 75%)
risk_thresholds = data[features].quantile([0.25, 0.75])

print("高風險和低風險閾值:")
print(risk_thresholds)

# 4. 創建新特徵來標記高風險/低風險顧客
def risk_category(row, feature, threshold_low, threshold_high):
    if row[feature] <= threshold_low:
        return 'Low'
    elif row[feature] >= threshold_high:
        return 'High'
    else:
        return 'Medium'

for feature in features:
    data[f'{feature}_Risk'] = data.apply(risk_category, axis=1, 
                                         feature=feature, 
                                         threshold_low=risk_thresholds.loc[0.25, feature], 
                                         threshold_high=risk_thresholds.loc[0.75, feature])

# 5. 將這些新創建的特徵進行One-Hot編碼，以便與機器學習模型兼容
risk_features = [f'{feature}_Risk' for feature in features]
data = pd.get_dummies(data, columns=risk_features, drop_first=True)

# 6. 選擇需要的訓練特徵（原始特徵 + 風險特徵）
selected_features = features + [col for col in data.columns if 'Risk' in col]
X = data[selected_features]
y = data['Attrition_Flag']

# 數據標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割訓練和測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. 使用K-fold進行交叉驗證和模型構建
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 構建XGBoost模型
xgb_model = XGBClassifier(learning_rate=0.1, max_depth=6, n_estimators=100, eval_metric='logloss', use_label_encoder=False)

# 使用 K-fold 進行交叉驗證評估模型
cross_val_scores = cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring='accuracy')

print(f"K-fold 交叉驗證準確率: {cross_val_scores}")
print(f"交叉驗證平均準確率: {cross_val_scores.mean():.2f}")

# 8. 訓練最終模型
xgb_model.fit(X_train, y_train)

# 9. 預測測試集
y_pred = xgb_model.predict(X_test)

# 10. 模型的準確性評估
accuracy = accuracy_score(y_test, y_pred)
print(f"測試集準確率: {accuracy:.2f}")

# 查看分類報告
report = classification_report(y_test, y_pred)
print(report)

# 11. 特徵重要性視覺化
plt.figure(figsize=(10, 6))
importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': xgb_model.feature_importances_
})
importance_df = importance_df.sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance in Predicting Customer Churn Risk")
plt.show()
