from sklearn.datasets import load_iris
import pandas as pd

# 載入資料集
iris = load_iris()

# 轉換成 DataFrame 格式方便觀看（結構化資料表格）
# 將 Iris dataset 的 data 屬性 (features) 指派給變數 X (feature matrix)
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# 將 Iris dataset 的 target 屬性 (class label) 指派給變數 y (target array) 
y = pd.DataFrame(data=iris.target, columns=['class label'])
# 沿著行 (column) 的方向串接 X 與 y，組成完整的樣本 (samples)
iris = pd.concat([X, y], axis=1)

# 印出樣本 (pandas 預設會將超過特定筆數的資料進行折疊隱藏，只顯示頭尾各 5 筆)
# 手動覆寫最大顯示列數設定（None 代表不限制，完整顯示）
#pd.set_option('display.max_rows', None)
print(iris)

