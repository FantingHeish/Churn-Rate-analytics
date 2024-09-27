import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加載數據
file_path = '/Users/fangting/Desktop/Code/BankChurners copy.csv'  # 替換為你的實際路徑
data = pd.read_csv(file_path)

# Y軸選定的變數 (注意，這些變數在X軸中不應重複)
y_features = ['Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Revolving_Bal', 
              'Total_Ct_Chng_Q4_Q1', 'Total_Amt_Chng_Q4_Q1', 
              'Avg_Utilization_Ratio', 'Avg_Open_To_Buy', 'Credit_Limit']

# X軸選定的變數，移除與Y軸重複的變數
x_features = ['Customer_Age', 'Gender', 'Dependent_count',
             'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 
            'Contacts_Count_12_mon']

# 將類別變數進行 One-Hot 編碼，確保所有變數為數值型
data_encoded = pd.get_dummies(data, drop_first=True)

# 確認 One-Hot 編碼後的 X 變數
# 因為 'Gender' 被分成 'Gender_M', 'Gender_F'，因此我們需要用 get_dummies 後的列
x_features_encoded = [col for col in data_encoded.columns if col in x_features or col.startswith('Gender')]

# 檢查是否存在任何 NaN 值，並進行填充（如果有必要）
data_encoded = data_encoded.fillna(0)

# 計算所有 X 變數和 Y 變數的相關性矩陣
correlation_matrix = data_encoded[y_features + x_features_encoded].corr().loc[y_features, x_features_encoded]

# 視覺化相關矩陣，去掉方框中的小字
plt.figure(figsize=(15, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='RdYlBu', linewidths=0.5, center=0, square=False, cbar_kws={"shrink": 0.8})

# 設置標題和顯示
plt.title('Correlation Matrix of Selected Y Features with All X Features (One-Hot Encoded)')
plt.xticks(rotation=45, ha='right', fontsize=12, style='italic')
plt.tight_layout()
plt.show()
