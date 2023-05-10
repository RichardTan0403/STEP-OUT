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

Y = dataset['Result']
X = dataset


cols = X.columns
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
X = ms.fit_transform(X)
X = pd.DataFrame(X, columns=[cols])
X

from sklearn.cluster import KMeans
cs = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init= 10, random_state = 5)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)

plt.plot(range(1,11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('CS')
plt.show()

kmeans = KMeans(n_clusters=2, random_state=5)
kmeans.fit(X)
labels = kmeans.labels_
# labels = pd.DataFrame(labels)
# labels.value_counts()

correct_labels = sum(Y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, Y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(Y.size)))