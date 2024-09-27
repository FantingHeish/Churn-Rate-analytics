import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 1. 加載數據
file_path = '/Users/fangting/Desktop/Code/BankChurners copy.csv'  # 替換為你的實際路徑
data = pd.read_csv(file_path)

# 確認你選擇的變數在數據中
selected_features = ['Avg_Utilization_Ratio', 'Total_Revolving_Bal']

# 移除缺失值，確保聚類數據沒有 NaN
data_clean = data[selected_features].dropna()

# KMeans 聚類（設置3個群體）
kmeans = KMeans(n_clusters=3, random_state=42)

# 聚類並將結果存到數據框中
data_clean['Risk_Group'] = kmeans.fit_predict(data_clean)

# 可視化聚類結果
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Avg_Utilization_Ratio', y='Total_Revolving_Bal', hue='Risk_Group', data=data_clean, palette='Set2')
plt.title('根據平均使用率和總循環餘額進行風險分群')
plt.show()
