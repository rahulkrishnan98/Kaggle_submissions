import csv
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt
filename="train.csv"
columns=[]
rows=[]
#converting age to float and int
def convertStr(s):
    """Convert string to either int or float."""
    try:
        ret = int(s)
    except ValueError:
        #Try float.
        ret = float(s)
    return ret
#reading the dataset
with open(filename, 'r') as csvfile:
    csvreader=csv.reader(csvfile)

    columns=next(csvreader)

    for row in csvreader:
        rows.append(row)
#taking the important columns for features
p_id=[]
p_class=[]
age=[]
sex=[]
output=[]
for i in rows:
    p_id.append(i[0])
    p_class.append(i[2])
    age.append(i[5])
    sex.append(i[4])
    output.append(i[1])

#Age has missing values, I fill -1 for all those

for i in range(len(age)):
    if not age[i]:
        age[i]=-1

for i in range(len(age)):
    age[i]=convertStr(age[i])
#converting the age to classes of values
age1=[]
for i in range(len(age)):
    if(age[i]==-1):
        age1.append('missing')
    elif(age[i]>=0 and age[i]<5):
        age1.append('baby')
    elif(age[i]>=5 and age[i]<12):
        age1.append('child')
    elif(age[i]>=12 and age[i]<18):
        age1.append('Teen')
    elif(age[i]>=18 and age[i]<35):
        age1.append('Young adult')
    elif(age[i]>=35 and age[i]<60):
        age1.append('Adult')
    else:
        age1.append('Aged')
#converting sex
for i in range(len(p_id)):
    if(sex[i]=='male'):
        sex[i]=1
    else:
        sex[i]=0
p_id=np.array(p_id, dtype=np.float32)
p_class=np.array(p_class, dtype=np.float32)
sex=np.array(sex, dtype=np.float32)
age=np.array(age, dtype=np.float32)
x=np.column_stack((p_class,age,sex))
y=output[:]



#for i in range(len(p_id)):
#    x.append([np.array(p_id[i]),np.array(p_class),np.array(sex),np.array(age)])

#The logistic regression model

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()

logisticRegr.fit(x, y)



#converting my test
columns1=[]
rows1=[]
with open("test.csv", 'r') as csvfile:
    csvreader=csv.reader(csvfile)

    columns1=next(csvreader)

    for row in csvreader:
        rows1.append(row)
p_id1=[]
p_class1=[]
sex1=[]
age1=[]
for i in range(len(rows1)):
    p_id1.append(rows1[i][0])
    p_class1.append(rows1[i][1])
    sex1.append(rows1[i][3])
    age1.append(rows1[i][4])
for i in range(len(age1)):
    if not age1[i]:
        age1[i]=-1

for i in range(len(age1)):
    age1[i]=convertStr(age1[i])
for i in range(len(rows1)):
    if(sex1[i]=='male'):
        sex1[i]=1
    else:
        sex1[i]=0
p_class1=np.array(p_class1, dtype=np.float32)
sex1=np.array(sex1, dtype=np.float32)
age1=np.array(age1, dtype=np.float32)
x1=np.column_stack((p_class1,age1,sex1))
predictions = logisticRegr.predict(x1)

wri=[['PassengerId','Survived']]
for i in range(len(predictions)):
    wri.append([p_id1[i],predictions[i]])


myFile = open('submit.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(wri)

