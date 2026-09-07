import pandas as pd

s = 'iris.data'
featureNames = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "class label"]
iris = pd.read_csv(s, names=featureNames, encoding='utf-8')    
    
print(iris)

