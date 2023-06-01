import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randint(0,101,size=(316800, 1)),columns=['Result'])
a = df.Result.tolist()
ColdList =[]
for x in a:
    if a[x] < 61:
        ColdList.append(0)
    else:
        ColdList.append(1)

Cold = pd.DataFrame(data = ColdList, columns=['Cold'])
FeverList = []
for x in ColdList:
    if ColdList[x] == 1:
        rand = np.random.randint(0, 101)
        if rand > 40:
            FeverList.append(1)
        else: 
            FeverList.append(0)
    if ColdList[x]==0:
        rand = np.random.randint(0,101)
        if rand > 65:
            FeverList.append(1)
        else:
            FeverList.append(0)

Fever = pd.DataFrame(data = FeverList, columns =['Fever'])

FluList = []
for x in ColdList:
    if ColdList[x] == 1:
        rand = np.random.randint(0, 101)
        if rand > 30:
            FluList.append(1)
        else: 
            FluList.append(0)
    if ColdList[x]==0:
        rand = np.random.randint(0,101)
        if rand > 65:
            FluList.append(1)
        else:
            FluList.append(0)

Flu = pd.DataFrame(data = FluList, columns = ['Flu'])

CoughingList= []
for x in FluList:
    if FluList[x] == 1:
        rand = np.random.randint(0, 101)
        if rand > 41:
            CoughingList.append(1)
        else: 
            CoughingList.append(0)
    if FluList[x]==0:
        rand = np.random.randint(0,101)
        if rand > 65:
            CoughingList.append(1)
        else:
            CoughingList.append(0)

Coughing = pd.DataFrame(data = CoughingList, columns = ['Coughing'])

SoreList = []
for x in FluList:
    if (FluList[x] == 1) & (CoughingList[x] == 1):
        rand = np.random.randint(0,101)
        if rand > 41:
            SoreList.append(1)
        else:
            SoreList.append(0)
    if (FluList[x] == 1)| (CoughingList[x] == 1):
        rand = np.random.randint(0,101)
        if rand >51:
            SoreList.append(1)
        else:
            SoreList.append(0)
    if (FluList[x] == 0) & (CoughingList == 0):
        rand = np.random.randint(0,101)
        if rand > 61:
            SoreList.append(1)
        else :
            SoreList.append(0)

Sore_Throat = pd.DataFrame(data = SoreList, columns = ['Sore_Throat'])

df = pd.DataFrame(np.random.randint(0,101,size=(316800, 1)),columns=['Result'])
a = df.Result.tolist()
DiarrheaList =[]
for x in a:
    if a[x] < 71:
        DiarrheaList.append(0)
    else:
        DiarrheaList.append(1)

Diarrhea = pd.DataFrame(data= DiarrheaList, columns= ['Diarrhea'])
Diarrhea.value_counts()

FatigueList = []
for x in FeverList:
    if (FeverList[x] == 1) & (CoughingList[x] == 1) & (FluList[x] == 1) & (DiarrheaList[x] == 1):
        rand = np.random.randint(0,101)
        if rand > 27:
            FatigueList.append(1)
        else:
            FatigueList.append(0)
    if (FeverList[x] == 1)| (DiarrheaList[x] == 1):
        rand = np.random.randint(0,101)
        if rand > 41:
            FatigueList.append(1)
        else:
            FatigueList.append(0)
    if (CoughingList[x] == 1) |(FluList[x] == 1):
        rand = np.random.randint(0,101)
        if rand > 61:
            FatigueList.append(1)
        else:
            FatigueList.append(0)
    if (FeverList[x] == 0) & (CoughingList == 0) & (FluList[x] == 0) & (DiarrheaList[x] == 0):
        rand = np.random.randint(0,101)
        if rand > 71:
            FatigueList.append(1)
        else :
            FatigueList.append(0)

Fatigue = pd.DataFrame(data= FatigueList, columns= ['Fatigue'])
Fatigue.value_counts()

df1 = pd.DataFrame(np.random.randint(0,101,size=(316800, 1)),columns=['Result'])
a = df1.Result.tolist()
MuscleList =[]
for x in a:
    if a[x] < 71:
        MuscleList.append(0)
    else:
        MuscleList.append(1)

Muscle_Ache = pd.DataFrame(data= MuscleList, columns= ['MuscleAche'])
Muscle_Ache.value_counts()

df = pd.DataFrame(np.random.randint(0,101,size=(316800, 1)),columns=['Result'])
a = df.Result.tolist()
AllergiesList =[]
for x in a:
    if a[x] < 71:
        AllergiesList.append(0)
    else:
        AllergiesList.append(1)

Allergies= pd.DataFrame(data= AllergiesList, columns= ['Allergies'])
Allergies.value_counts()

result = pd.concat([Allergies, Flu, Coughing, Diarrhea, Fatigue, Fever, Muscle_Ache, Sore_Throat, Cold], axis = 1)
result.to_csv('symptoms.csv', index=False)

df = pd.DataFrame(np.random.randint(0,101,size=(316800, 1)),columns=['Result'])
a = df.Result.tolist()
OXygenList =[]
for x in a:
    if a[x] < 21:
        OXygenList.append(np.random.randint(90,101))
    else:
        OXygenList.append(np.random.randint(95,101))
Oxygen = pd.DataFrame(data=OXygenList, columns=['Oxygen_Level'])
Oxygen.value_counts()
Oxygen.to_csv('O.csv', index=False)

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