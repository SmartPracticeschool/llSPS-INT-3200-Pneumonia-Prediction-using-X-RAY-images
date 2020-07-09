#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


# In[2]:


#initialize the model
model=Sequential()


# In[3]:


model.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))


# In[4]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[5]:


model.add(Flatten())


# In[6]:


model.add(Dense(output_dim=128,init='uniform',activation='relu'))


# In[7]:


model.add(Dense(output_dim=1,activation='sigmoid',init='uniform'))


# In[8]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)


# In[9]:


x_train=train_datagen.flow_from_directory(r'C:\DATASET\Train set',target_size=(64,64),batch_size=32,class_mode='binary')
x_test=test_datagen.flow_from_directory(r'C:\DATASET\Test set',target_size=(64,64),batch_size=32,class_mode='binary')


# In[10]:


print(x_train.class_indices)


# In[11]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[12]:


model.fit_generator(x_train,samples_per_epoch=44,epochs=10,validation_data=x_test,nb_val_samples=19)


# In[13]:


model.save('mymodel.h5')


# In[14]:


from keras.models import load_model
from keras.preprocessing import image


# In[15]:


model=load_model('mymodel.h5')


# In[16]:


img=image.load_img(r'C:\Users\DEESHMA REDDY\Desktop\normal.jpeg',target_size=(64,64))


# In[17]:


import numpy as np


# In[18]:


x=image.img_to_array(img)
x = np.expand_dims(x,axis=0)


# In[19]:


pred=model.predict_classes(x)


# In[20]:


pred


# In[ ]:





# In[ ]:




