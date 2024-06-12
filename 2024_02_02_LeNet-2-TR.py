#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[ ]:


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
    df = df5
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


# In[ ]:


inputs_np = np.array(inputs)
outputs_np = np.array(outputs)


# In[ ]:


index_oneshot = np.arange(0,280,40)
index_twoshot = np.arange(0,280,20)
index_threeshot = np.sort(np.concatenate((index_oneshot+1,index_twoshot)))
index_fourshot = np.sort(np.concatenate((index_twoshot+1,index_twoshot)))
index_fiveshot = np.sort(np.concatenate((index_oneshot+2,index_fourshot)))


# In[ ]:


index_test = np.array([0])
for i in range(5,20):
    temp  = np.arange(0,280,20)+i
    index_test = np.sort(np.concatenate((temp,index_test)))
index_test = index_test[1:]


# In[ ]:


base_model = keras.models.load_model('base_model2.keras')


# In[ ]:


# X_rem, X_test, y_rem, y_test = train_test_split(inputs_np,outputs_np, train_size=0.1, stratify = outputs_np)
# X_train, X_valid, y_train, y_valid = train_test_split(inputs_np[index_twoshot],outputs_np[index_twoshot], train_size=0.5, stratify = outputs_np[index_twoshot])


# In[ ]:


X_train = inputs_np[index_fourshot]
y_train = outputs_np[index_fourshot]

X_test = inputs_np[index_test]
y_test = outputs_np[index_test]


# In[ ]:


predictions = base_model.predict(X_test)


# In[ ]:


predict_class = np.argmax(predictions, axis=1)
ground_truth = np.argmax(y_test, axis=1)


# In[ ]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
sns.set()
f,ax=plt.subplots()
y_true = ground_truth
y_pred = predict_class
C2= confusion_matrix(y_true, y_pred)
sns.heatmap(C2,annot=True,ax=ax)
ax.set_title('confusion matrix') 
ax.set_xlabel('predict') 
ax.set_ylabel('true') 
plt.savefig('test.png', dpi = 300)


# In[ ]:


scores = base_model.evaluate(X_test,y_test,verbose = 0)
print(scores)


# In[ ]:


new_model = base_model


# In[ ]:


def reinitialize_layer(model, initializer, layer_name):
    layer = model.get_layer(layer_name)    
    layer.set_weights([initializer(shape=w.shape) for w in layer.get_weights()])
    
initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
reinitialize_layer(new_model, initializer, "dense_2") 


# In[ ]:


# build the model and train it
import tensorflow_addons as tfa

optimizers = [
    tf.keras.optimizers.Adam(learning_rate=1e-3),
    tf.keras.optimizers.Adam(learning_rate=1e-2)
]
optimizers_and_layers = [(optimizers[0], new_model.layers[0:-2]), (optimizers[1], new_model.layers[-1])]
optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
new_model.compile(optimizer= optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = new_model.fit(X_train, y_train,validation_data=(X_train, y_train), epochs=16, batch_size=3)


# In[ ]:


predictions2 = new_model.predict(X_test)
predict_class2 = np.argmax(predictions2, axis=1)
ground_truth = np.argmax(y_test, axis=1)

sns.set()
f,ax=plt.subplots()
y_true2 = ground_truth
y_pred2 = predict_class2
C22= confusion_matrix(y_true2, y_pred2)
sns.heatmap(C22,annot=True,ax=ax) 

ax.set_title('confusion matrix') 
ax.set_xlabel('predict')
ax.set_ylabel('true') 

plt.savefig('test2.png', dpi = 300)


# In[ ]:


scores = new_model.evaluate(X_test,y_test,verbose = 0)
print(scores)


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# #### 
