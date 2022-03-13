# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 18:42:39 2021

@author: Laraib Quamar 
Aligarh Muslim University 
"""


from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Add, Reshape,Cropping3D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
path="test200_data/*.dat"
count =0
file_list= glob.glob(path)
# print(file_list)
file_list.sort()
#print(file_list)


for data_file in file_list:
    df = pd.read_csv(data_file,
                 sep="\s+",
                 skiprows=1,
                 usecols=[2,3],
                 names=['UVel','Vvel'])


    #print(x)
    comb = np.array(df[['UVel','Vvel']])
    #print(u_vel)
    if count == 0:
        alldata = comb
    else:
        alldata = np.vstack((alldata,comb))
    count += 1
W=alldata.reshape(2000,379,129,2)
                                          

     

J=alldata.reshape(2000,48891,2)                                        
X = J[:,:,0]
Y = J[:,:,1]


X = np.array(X)
rows =2000 
cols = 48891

Umode= []
for i in range(0,cols):  
    XsumCol = 0 
    for j in range(0, rows):  
        XsumCol += X[j][i]
        #Uf0  = (sumCol/rows)
    print("Sum of " + str(i+1) +" column: " + str(XsumCol))
    Umode.append(XsumCol/2000)
 # 



Y =np.array(Y)
Vmode =[]
for i in range(0,cols):
    YsumCols = 0
    for j in range(0,rows):
        YsumCols += Y[j][i]
    Vmode.append(YsumCols/2000)
    
#print("Ylist",Vmode.shape)

x_num0=379; y_num0=129;
Uf0=Umode[0:x_num0*y_num0]
Vf0=Vmode[0:x_num0*y_num0]
Uf0=np.reshape(Uf0,[x_num0,y_num0])
Vf0=np.reshape(Vf0,[x_num0,y_num0])


for i in range(len(X)):
    W[i,:,:,0]=W[i,:,:,0]-Uf0
    W[i,:,:,1]=W[i,:,:,1]-Vf0

  
W=np.pad(W,((0,0),(3,2),(32,31),(0,0)),'constant')
                               

input_data = Input(shape=(384, 192, 2))
## Encoder
x = Conv2D(16, (3, 3), activation='tanh', padding='same')(input_data)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Reshape([6*3*4])(x)
encoded = Dense(2,activation='tanh')(x)
#encoder=Model(input_data,encoded)
#encoder.summary()

## Two variables
val1= Lambda(lambda x: x[:,0:1])(encoded)
val2= Lambda(lambda x: x[:,1:2])(encoded)
## Decoder 1
#decoder_input = Input(shape=(2), name="decoder_input")
x1 = Dense(6*3*4,activation='tanh')(val1)
x1 = Reshape([6,3,4])(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(4,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(16,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1d = Conv2D(2,(3,3),activation='linear',padding='same')(x1)
## Decoder 2
x2 = Dense(6*3*4,activation='tanh')(val2)
x2 = Reshape([6,3,4])(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(4,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(8,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(8,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(8,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(16,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2d = Conv2D(2,(3,3),activation='linear',padding='same')(x2)

decoded = Add()([x1d,x2d])
autoencoder = Model(input_data, decoded)



autoencoder = Model(input_data, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Check the network structure
autoencoder.summary()



an='./qwerty.hdf5'
model_cb= ModelCheckpoint(an, monitor='val_loss',save_best_only=True,verbose=1)
early_cb= EarlyStopping(monitor='val_loss', patience=25,verbose=1)
cb = [model_cb, early_cb]

X_train,X_test,y_train,y_test=train_test_split(W,W,test_size=0.3,random_state=1)

history=autoencoder.fit(W, W,
                epochs=2000,
                batch_size=10,
                shuffle=True,
                #validation_data=(X_test, y_test),
                callbacks=cb)
print("the value of loss for act fn tanh snapshot taken 2000 for MD-CNN")
#
#encoded_data=encoded.predict(W)
#print("the value of loss for act fn tanh snapshot taken 500 for MD-CNN")
decoded_data=autoencoder.predict(W)

R=decoded_data[:,3:382,32:161,:]
for i in range(len(R)):
    R[i,:,:,0]=R[i,:,:,0]+Uf0
    R[i,:,:,1]=R[i,:,:,1]+Vf0
R=R.reshape(2000,48891,2)



L = np.array(R)
T = L[:,:,0]
O = L[:,:,1]
print(T.shape)
print("zas",T)
print("nas",O)

print("maximum",L)

P= T.reshape(2000,48891)
S= O.reshape(2000,48891)

import os

# Determine incremented filename
i = 0
# os.path.exists(f"file_{i}.dat"):
for i in range(0,2000):
    
    
    if i<9:
        source_file = "test200_data//SNAP000{0}.dat".format(str(i+1))
    elif i<99:
        source_file = "test200_data//SNAP00{0}.dat".format(str(i+1))
    elif i<999:
        source_file = "test200_data//SNAP0{0}.dat".format(str(i+1))
    else: 
        source_file = "test200_data//SNAP{0}.dat".format(str(i+1))
    X = pd.read_csv(source_file,
                 sep="\s+",
                 skiprows=1,
                 usecols=[0],
                 names=['xx'])
    Y = pd.read_csv(source_file,
                 sep="\s+",
                 skiprows=1,
                 usecols=[1],
                 names=['yy'])
    #file = open(f"file_{i}.dat", "w")
# ... Do some processing ...
    content=pd.DataFrame({'ss':X.xx.values ,'zz': Y.yy.values ,'vv':P[i][:] ,'vv':S[i][:]})
    #file.write((content))
    file = "TANX2000mDcnnfile_{0}.csv".format(str(i))
    content.to_csv(file,index = False)

history.history.keys()
plt.plot(history.history["loss"])
plt.title("model_loss MD-CNN tanh")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train"],loc ="upper_left")
plt.show()
#%%
