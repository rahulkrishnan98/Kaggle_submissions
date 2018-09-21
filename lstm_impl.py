import csv
import numpy as np
filename="newdata.csv"
columns=[]
rows=[]

def convertStr(s):
    """Convert string to either int or float."""
    try:
        ret = int(s)
    except ValueError:
        #Try float.
        ret = float(s)
    return ret

with open(filename, 'r') as csvfile:
    csvreader=csv.reader(csvfile)

    columns=next(csvreader)

    for row in csvreader:
        rows.append(row)

x_train=[]
y_train=[]
seq_no=[]
shop_id=[]
item_id=[]
output=[]

for i in rows:
    seq_no.append(i[0])
    shop_id.append(i[1])
    item_id.append(i[2])
    output.append(convertStr(i[3]))
seq_no=np.array(seq_no, dtype=np.float32)
shop_id=np.array(shop_id, dtype=np.float32)
item_id=np.array(item_id, dtype=np.float32)

x_train=np.column_stack((seq_no,shop_id,item_id))
y_train=output[:]

x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))

#building the lstm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model_lstm = Sequential()
model_lstm.add(LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2])))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


history = model_lstm.fit(x_train, y_train, batch_size=5, nb_epoch=1,validation_data=(x_train,y_train), shuffle=True,verbose=2)

#prediction

columns1=[]
rows1=[]
with open("test.csv", 'r') as csvfile1:
    csvreader1=csv.reader(csvfile1)

    columns1=next(csvreader1)

    for row in csvreader1:
        rows1.append(row)
shop_id_res=[]
item_id_res=[]
seq_no_res=[34]*len(rows1)
for i in rows1:
    shop_id_res.append(i[1])
    item_id_res.append(i[2])
seq_no_res=np.array(seq_no_res, dtype=np.float32)
shop_id_res=np.array(shop_id_res, dtype=np.float32)
item_id_res=np.array(item_id_res, dtype=np.float32)

x_train_res=np.column_stack((seq_no_res,shop_id_res,item_id_res))
x_train_res = x_train_res.reshape((x_train_res.shape[0], 1, x_train_res.shape[1]))

y= model_lstm.predict(x_train_res)
wri=[['ID','item_cnt_month']]
for i in range(len(y)):
    wri.append([i, str(y[i][0])])

myFile = open('submissionlstm.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(wri)