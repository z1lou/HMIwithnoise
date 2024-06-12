import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam


GESTURES = ["g1","g2","g3","g4","g5","g6","g7"]
SAMPLES_PER_GESTURE = 100
NUM_GESTURES = len(GESTURES)
ONE_HOT_GESTURES = np.eye(NUM_GESTURES) 
NUM_SENSOR = 6 

inputs = [] 
outputs = [] 

scaler = MinMaxScaler()
for g_idx in range(NUM_GESTURES):
    g = GESTURES[g_idx]
    output = ONE_HOT_GESTURES[g_idx]
    df1 = pd.read_csv("subject1/" + g + ".csv",header = None)
    df2 = pd.read_csv("subject2/" + g + ".csv",header = None)
    df3 = pd.read_csv("subject3/" + g + ".csv",header = None)
    df4 = pd.read_csv("subject4/" + g + ".csv",header = None)
    df5 = pd.read_csv("subject5/" + g + ".csv",header = None)
    df6 = pd.read_csv("subject6/" + g + ".csv",header = None)
    df = pd.concat([df1,df2,df3,df4,df6])
    df_scaled = scaler.fit_transform(df) 
    df_scaled_DF = pd.DataFrame(df_scaled)  
    df_scaled_DF = df_scaled_DF.dropna()   
    num_recordings = int(df_scaled_DF.shape[0] / SAMPLES_PER_GESTURE)

    for i in range(num_recordings):
        sensorData = df_scaled_DF.iloc[i*SAMPLES_PER_GESTURE:(i+1)*SAMPLES_PER_GESTURE,:]
        sensorData_np = np.array(sensorData) 
        sensorData_np = np.reshape(sensorData_np,SAMPLES_PER_GESTURE*NUM_SENSOR)
        inputs.append(sensorData_np)
        outputs.append(output)

inputs_np = np.array(inputs)
outputs_np = np.array(outputs)

X_train, X_test, y_train, y_test = train_test_split(inputs_np,outputs_np, train_size=0.7, stratify= outputs_np)
#X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5,stratify= y_rem)

temp = np.array(X_train)
temp.shape
temp2 = np.array(y_train)
temp2.shape

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(600,1)))
model.add(tf.keras.layers.Conv1D(filters = 8,activation='relu', kernel_size = 3))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Conv1D(filters = 16,activation='relu', kernel_size = 5))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64, activation='relu')) 
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(NUM_GESTURES, activation='softmax')) 
optimizer = Adam(learning_rate=0.01)
model.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=50, batch_size=10)
model.save('base_model2.keras')

model_accuracy = np.array(history.history['accuracy'])
model_val_accuracy = np.array(history.history['val_accuracy'])
model_loss = np.array(history.history['loss'])
model_val_loss = np.array(history.history['val_loss'])
np.savetxt("acc.txt",model_accuracy)
np.savetxt("val_acc.txt",model_val_accuracy)
np.savetxt("loss.txt",model_loss)
np.savetxt("val_loss.txt",model_val_loss)

