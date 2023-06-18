#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D,BatchNormalization
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from glob import glob
from  PIL import Image  #load image nd resizing purpose
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[3]:


df=pd.read_csv("C:\\softnerve\\HAM10000_metadata.csv")
df


#  SEVEN CLASSES OF LESIONS:
#  Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec),
#  basal cell carcinoma (bcc),
#  benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl),
#  dermatofibroma (df), 
#  melanoma (mel), 
#  melanocytic nevi (nv) and 
#  vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).

# In[4]:


df['dx'].value_counts()


# In[5]:


ax=sns.countplot(x='dx',data=df)
plt.title("Categories of Pigmented Lesions")
plt.xticks(rotation=45)
plt.show()


# Here we can see that outof seven types,melanocytic nevi(nv) found to be more in numbers

# In[104]:


ax=sns.countplot(x='localization',data=df)
plt.title("Localisation area frequency")
plt.xticks(rotation=90)
plt.show()


# After analysing the back and lower extremity parts have higher accumulation of cancer cells whereas,ear and genital parts are less effected.

# In[105]:


sns.barplot(x='sex',y='age',data=df)
plt.xlabel("sex")
plt.ylabel("age")
plt.show()


# comparing to females,aged patients are more in males.The number of aged female patients is comparatively low.

# In[106]:


sns.set_style('whitegrid')
fig, axes = plt.subplots(figsize=(12,8))
ax = sns.histplot(data=df, x = 'age', hue = 'sex', multiple='stack')
plt.title('age histogram gender wiese')
plt.show()


# In[107]:


np.random.seed(42)
size=32


# In[108]:


#Label encoding on dx column
le=LabelEncoder()
le.fit(df['dx'])
print(list(le.classes_))


# In[109]:


df['label']=le.transform(df['dx'])
df['label']


# In[110]:


df


# In[111]:


#distribution of data into different classes
from sklearn.utils import resample

df_0=df[df['label']==0]
df_1=df[df['label']==1]
df_2=df[df['label']==2]
df_3=df[df['label']==3]
df_4=df[df['label']==4]
df_5=df[df['label']==5]
df_6=df[df['label']==6]

#using RESAMPLE upscail or downscail samples
df_0_balanced=resample(df_0,replace=True,n_samples=500,random_state=42)
df_1_balanced=resample(df_1,replace=True,n_samples=500,random_state=42)
df_2_balanced=resample(df_2,replace=True,n_samples=500,random_state=42)
df_3_balanced=resample(df_3,replace=True,n_samples=500,random_state=42)
df_4_balanced=resample(df_4,replace=True,n_samples=500,random_state=42)
df_5_balanced=resample(df_5,replace=True,n_samples=500,random_state=42)
df_6_balanced=resample(df_6,replace=True,n_samples=500,random_state=42)
#resampled and balanced dataframe
df_balanced=pd.concat([df_0_balanced,df_1_balanced,df_2_balanced,df_3_balanced,df_4_balanced,df_5_balanced,df_6_balanced])
df_balanced


# Now each class of lesions have equal amount of elements ie,a balanced dataset for training

# In[112]:


#fetchhing images and combining them into one folder
image_path={os.path.splitext(os.path.basename(x))[0]:x for x in glob(os.path.join('C:\\softnerve\\HAM100000','*','*.jpg'))}


# In[113]:


df_balanced['path']=df['image_id'].map(image_path.get)
df_balanced['image']=df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((size,size))))
df_balanced


# In[114]:


#plotting the images
no_samples=5
fig,m_axs=plt.subplots(7,no_samples,figsize=(4*no_samples,3*7))
for n_axs,(type_name,type_rows) in zip(m_axs,df_balanced.sort_values(['dx']).groupby('dx')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(no_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')


# In[116]:


#feature and target selection
x=np.asarray(df_balanced['image'].tolist())   #array of uint8
x=x/255  #max pixel is255   #array of float64
y=df_balanced['label']   #series
# for a multiclass classification structure should be changed to categorical values
y_cat=to_categorical(y,num_classes=7)

#splitting and training data
x_train,x_test,y_train,y_test=train_test_split(x,y_cat,test_size=0.25,random_state=42)


# In[117]:


num_classes=7
model=Sequential()
model.add(Conv2D(256,(3,3),activation='relu',input_shape=(size,size,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(32))
model.add(Dense(7,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['acc'])


# In[119]:


history=model.fit(x_train,y_train,epochs=60,batch_size=16,validation_data=(x_test,y_test),verbose=2)


# In[120]:


score=model.evaluate(x_test,y_test)
print("test accuracy",score[1])


# In[121]:


loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'y',label='Training_loss')
plt.plot(epochs,val_loss,'r',label='Validation_loss')
plt.title("Training and Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

