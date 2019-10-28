
# coding: utf-8

# In[1]:


import numpy as np
import pandas
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

import keras
from keras.layers import Activation, Input, Embedding, LSTM, Dense, Lambda, GaussianNoise, concatenate
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras import backend as K
#from keras.layers import Dense, Merge
from keras.optimizers import SGD, Adam, RMSprop
from keras.constraints import max_norm
from keras.layers import MaxPooling1D, Dropout, Dense, Flatten, Activation, Conv1D
from keras.models import Sequential
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy
import matplotlib.pyplot as plt


# In[2]:


seed = 7
np.random.seed(seed)


# In[3]:


dataframe = pandas.read_csv("Beats_Ext_FN.csv", header=0)
#dataframe = dataframe[1::2]
dataset = dataframe.values
X = dataset[:,0:2].astype(float)
Y = dataset[:,3]


# In[4]:


encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

#split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y)

# convert integers to dummy variables (i.e. one hot encoded)
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)


# In[5]:


X_train.shape


# In[6]:


X_test.shape


# In[7]:


# Teacher model

nb_classes = 3
input_shape = X_train.shape # Input shape of each image

# Hyperparameters
nb_filters = 64 # number of convolutional filters to use
pool_size = 2 # size of pooling area for max pooling
kernel_size = 2 # convolution kernel size

teacher = Sequential()

X_train = np.expand_dims(X_train, axis=2) 
X_test = np.expand_dims(X_test, axis=2)

teacher.add(Conv1D(32, kernel_size=2,
                 activation='relu',
                 input_shape=(2,1)))
teacher.add(Conv1D(64, 1, activation='relu'))
teacher.add(MaxPooling1D(pool_size=1))

teacher.add(Dropout(0.25)) # For reguralization

teacher.add(Flatten())
teacher.add(Dense(128, activation='relu'))
teacher.add(Dropout(0.5)) # For reguralization

teacher.add(Dense(nb_classes))
teacher.add(Activation('softmax')) # Note that we add a normal softmax layer to begin with

teacher.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

print(teacher.summary())


# In[8]:


# Student model that is stand-alone. We will evaluate its accuracy compared to a teacher trained student model

student = Sequential()
#X_train = np.expand_dims(X_train, axis=2) 
#X_test = np.expand_dims(X_test, axis=2)
student.add(Flatten(input_shape=(2,1)))
student.add(Dense(32, activation='relu'))
student.add(Dropout(0.2))
student.add(Dense(nb_classes))
student.add(Activation('softmax'))

#sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
student.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

student.summary()


# In[9]:


import os
import psutil
pid = os.getpid()
py = psutil.Process(pid)
memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use:', memoryUse)


# In[10]:


# Train the teacher model as usual
epochs = 30
batch_size = 3

import timeit
start = timeit.default_timer()
history=teacher.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
stop = timeit.default_timer()
print('Time: ', stop - start)  
memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use:', memoryUse)


# In[11]:


import timeit
start = timeit.default_timer()
y_pred = teacher.predict(X_test)
stop = timeit.default_timer()
print('Time: ', stop - start)


# In[12]:


plt.rcParams["font.family"] = "Times New Roman"
plt.plot(history.history['acc'])
plt.title('Teacher Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.plot(history.history['loss'])
plt.title('Teacher Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


# In[13]:


# Raise the temperature of teacher model and gather the soft targets

# Set a tempature value
temp = 7

#Collect the logits from the previous layer output and store it in a different model
teacher_WO_Softmax = Model(teacher.input, teacher.get_layer('dense_2').output)


# In[14]:


# Define a manual softmax function
def softmax(x):
    return np.exp(x)/(np.exp(x).sum())


# In[15]:


#prepare the soft targets and target data for student to be trained upon
teacher_train_logits = teacher_WO_Softmax.predict(X_train)
teacher_test_logits = teacher_WO_Softmax.predict(X_test) # This model directly gives the logits ( see the teacher_WO_softmax model above)

# Perform a manual softmax at raised temperature
train_logits_T = teacher_train_logits/temp
test_logits_T = teacher_test_logits / temp 

Y_train_soft = softmax(train_logits_T)
Y_test_soft = softmax(test_logits_T)

# Concatenate so that this becomes a 10 + 10 dimensional vector
Y_train_new = np.concatenate([Y_train, Y_train_soft], axis=1)
Y_test_new =  np.concatenate([Y_test, Y_test_soft], axis =1)


# In[16]:


# Remove the softmax layer from the student network
student.layers.pop()

# Now collect the logits from the last layer
logits = student.layers[-1].output # This is going to be a tensor. And hence it needs to pass through a Activation layer
probs = Activation('softmax')(logits)

# softed probabilities at raised temperature
logits_T = Lambda(lambda x: x / temp)(logits)
probs_T = Activation('softmax')(logits_T)

output = concatenate([probs, probs_T])

# This is our new student model
student = Model(student.input, output)

student.summary()


# In[17]:


# This will be a teacher trained student model. 
# --> This uses a knowledge distillation loss function

# Declare knowledge distillation loss
def knowledge_distillation_loss(y_true, y_pred, alpha):

    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    y_true, y_true_softs = y_true[: , :nb_classes], y_true[: , nb_classes:]
    
    y_pred, y_pred_softs = y_pred[: , :nb_classes], y_pred[: , nb_classes:]
    
    loss = alpha*logloss(y_true,y_pred) + logloss(y_true_softs, y_pred_softs)
    
    return loss

# For testing use regular output probabilities - without temperature
def acc(y_true, y_pred):
    y_true = y_true[:, :nb_classes]
    y_pred = y_pred[:, :nb_classes]
    return categorical_accuracy(y_true, y_pred)

student.compile(
    #optimizer=optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True),
    optimizer='adadelta',
    loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, 0.1),
    #loss='categorical_crossentropy',
    metrics=[acc] )


# In[18]:


import timeit
start = timeit.default_timer()
history=student.fit(X_train, Y_train_new,
          batch_size=3,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test_new))
stop = timeit.default_timer()
print('Time: ', stop - start)


# In[19]:


import os
import psutil
pid = os.getpid()
py = psutil.Process(pid)
memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use:', memoryUse)


# In[20]:


import timeit
start = timeit.default_timer()
y_pred = student.predict(X_test)
stop = timeit.default_timer()
print('Time: ', stop - start)


# In[21]:


plt.rcParams["font.family"] = "Times New Roman"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Teacher-trained Student Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Teacher-trained Student Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Test'],loc='upper right')
plt.show()


# In[22]:


# This is a standalone student model (same number of layers as original student model) trained on same data
# for comparing it with teacher trained student.

n_student = Sequential()
n_student.add(Flatten(input_shape=(2,1)))
n_student.add(Dense(32, activation='relu'))
n_student.add(Dropout(0.2))
n_student.add(Dense(nb_classes))
n_student.add(Activation('softmax'))

#sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
n_student.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


# In[23]:


import timeit
start = timeit.default_timer()
history=n_student.fit(X_train, Y_train,
          batch_size=3,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test))
stop = timeit.default_timer()
print('Time: ', stop - start)


# In[24]:


plt.rcParams["font.family"] = "Times New Roman"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Standalone Student Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Standalone Student Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Test'],loc='upper right')
plt.show()


# In[25]:


import os
import psutil
pid = os.getpid()
py = psutil.Process(pid)
memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use:', memoryUse)


# In[26]:


import timeit
start = timeit.default_timer()
y_pred = n_student.predict(X_test)
stop = timeit.default_timer()
print('Time: ', stop - start)

