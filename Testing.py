import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

dataset = pd.read_csv("./Machine_Learning/Final_Sample.csv")
dataset = dataset.drop('Gender', axis = 1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
pd.cut(dataset.Age, bins =[16, 32, 60, 79], labels= [0,1,2])
pd.cut(dataset.Systolic_bp , bins = [70, 90, 120, 140, 190], labels = [0,1,2,3])
pd.cut(dataset.Diastolic_Bp, bins = [40, 60, 80,90, 100], labels =  [0,1,2,3])
pd.cut(dataset.Oxygen_Level, bins = [92, 94, 96, 100], labels = [0,1,2])

Y = dataset['Result']
X = dataset.drop('Result', axis = 1)

cols = X.columns
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
X = ms.fit_transform(X)
X = pd.DataFrame(X, columns=[cols])

from sklearn.model_selection import train_test_split as tts
Trained_X,Tested_X,Trained_Y,Tested_Y = tts(X,Y,test_size=0.2,random_state=5)

# from sklearn.svm import SVC
# model = SVC()
# from sklearn.model_selection import GridSearchCV
# param_grid = {'C': [0.1, 1, 10, 100, 1000], 
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'kernel': ['rbf']} 
# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
# grid.fit(Trained_X,Trained_Y)