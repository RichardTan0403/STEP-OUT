import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("./Machine_Learning/Sample.csv")
dataset.shape
dataset.isna().sum()
dataset.duplicated().sum()
dataset.describe()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['Gender'] = le.fit_transform(dataset['Gender'])
dataset.head()
pd.cut(dataset.Age, bins =[16, 32, 60, 79], labels= ['Young','Middle_Age', 'Old'])
pd.cut(dataset.Systolic_bp , bins = [70, 90, 120, 140, 190], labels = ['Low','Optimal','High Normal', 'Hypertension'])
pd.cut(dataset.diastolic_bp, bins = [40, 60, 80,90, 100], labels =  ['Low','Optimal','High Normal', 'Hypertension'])
pd.cut(dataset.Oxygen_Level, bins = [80,92, 94, 96, 100], labels = ['Extreme low', 'Low', 'Intermediate','Ideal'])

Y = dataset['Result'].array.reshape(-1,1)
X = dataset.drop('Result', axis = 1)

dataset['Result'].value_counts()


cols = X.columns
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
X = ms.fit_transform(X)
X = pd.DataFrame(X, columns=[cols])

sns.heatmap(dataset.corr(),vmin = -1, vmax = 1, annot= True)
plt.show()

from sklearn.model_selection import train_test_split as tts
Trained_X,Tested_X,Trained_Y,Tested_Y = tts(X,Y,test_size=0.2,random_state=5)

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
rfc = RFC()
rfc.fit(Trained_X, Trained_Y)

y_pred4 = rfc.predict(Tested_X)
acc = accuracy_score(Tested_Y, y_pred4)
print('Accuracy of Random Forest: ', acc*100, '%')

from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(Tested_Y, y_pred4)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
cm_display.plot()
plt.show()