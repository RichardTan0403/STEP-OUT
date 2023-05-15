import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randint(0,101,size=(316800, 1)),columns=['Result'])
a = df.Result.tolist()
b =[]
for x in a:
    if a[x] < 81:
        b.append(0)
    else:
        b.append(1)

b = pd.DataFrame(data = b, columns=['Cold'])
b.head()
b.value_counts()

b.to_csv('symptoms.csv', index=False)

# from sklearn.cluster import KMeans
# cs = []
# for i in range(1,11):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init= 10, random_state = 5)
#     kmeans.fit(X)
#     cs.append(kmeans.inertia_)

# plt.plot(range(1,11), cs)
# plt.title('The Elbow Method')
# plt.xlabel('Number of Clusters')
# plt.ylabel('CS')
# plt.show()

# kmeans = KMeans(n_clusters=2, random_state=5)
# kmeans.fit(Trained_X, Trained_Y)
# kmeans.predict(Tested_Y)
# labels = kmeans.labels_
# # labels = pd.DataFrame(labels)
# # labels.value_counts()

# correct_labels = sum(Tested_Y == labels)
# print("Result: %d out of %d samples were correctly labeled." % (correct_labels, Tested_Y.size))
# print('Accuracy score: {0:0.2f}'. format(correct_labels/float(Y.size)))