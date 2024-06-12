import tensorflow as tf
import pandas as pd
from keras import metrics
from keras import losses
from keras import optimizers
import numpy as np

# total.to_csv('combined_data.csv', index=False)

total = pd.read_csv('combined_data.csv')
rows = 7
cols = 10
answer = [[0 for j in range(cols)] for i in range(rows)]
# print(total)

독립 = total[['weekday', 'visitors', 'averagetemp','lowesttemp','highesttemp']]
종속 = total[['0001_salesVol', '0002_salesVol', '0003_salesVol', '0004_salesVol', '0005_salesVol', '0006_salesVol', '0007_salesVol', '0008_salesVol', '0009_salesVol', '0010_salesVol']]

X = tf.keras.layers.Input(shape=[5])
Y = tf.keras.layers.Dense(10)(X)
model = tf.keras.models.Model(X, Y)

model.compile( 
   loss = 'mean_squared_error', 
   metrics = 'accuracy',
)

model.fit(독립, 종속, epochs=10000, batch_size=32)

answer[0] = np.array(model.predict([[7, 58368, 22.2, 18.4, 27.5]]))
answer[1] = np.array(model.predict([[1, 58055, 22, 19.2, 26]]))
answer[2] = np.array(model.predict([[2, 48855, 20.6, 17.5, 25]]))
answer[3] = np.array(model.predict([[3, 58473, 22.7, 17.7, 28.3]]))
answer[4] = np.array(model.predict([[4, 56638, 23.7, 19.1, 28.9]]))
answer[5] = np.array(model.predict([[5, 47958, 24.2, 18.9, 30.2]]))
answer[6] = np.array(model.predict([[6, 41583, 25.9, 20.8, 32.6]]))

for i in range(rows):
    answer[i] = answer[i].astype(int)
    print(answer[i],sep="\n")