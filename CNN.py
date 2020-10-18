import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import urllib.request

from sklearn.model_selection import train_test_split
from IPython.core import history

dataset_url ="https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/cat.npy"
data_dir = tf.keras.utils.get_file(
    origin=dataset_url, fname='flower_photos', untar=True)
dataset_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/dog.npy"
data_dir = tf.keras.utils.get_file(
    origin=dataset_url, fname='flower_photos', untar=True)

cat = np.load('cat.npy', encoding='bytes', allow_pickle=True)
dog = np.load('dog.npy', encoding='bytes', allow_pickle=True)

dog.shape
cat.shape

fig=plt.figure(figsize=(10, 10))
columns = 5
rows = 1

for i in range(1, columns * rows + 1):
    img = dog[i].reshape(28, 28)

    fig.add_subplot(rows,columns, i)

    plt.imshow(img)

plt.show()

sample_size = 60000
X = np.concatenate((dog[:sample_size],cat[:sample_size]))

X = X.reshape(-1,28,28,1)/255.0
X.shape

Y = np.zeros(2*sample_size)
Y[sample_size:]=1.0
Y.shape

train_x,test_x,train_y,test_y=train_test_split(X,Y,random_state=41,test_size=0.3)
print(train_x.shape)
print(test_x.shape)
train_y=tf.keras.utils.to_categorical(train_y)
test_y=tf.keras.utils.to_categorical(test_y)

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),input_shape=(28,28,1),activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(2,activation='softmax')
])

tf.keras.utils.plot_model(model,show_shapes=True,rankdir='LR')
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

history=model.fit(x=train_x,y=train_y,batch_size=128,epochs=20,verbose=1,validation_split=0.2)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.legend(['train','validation'],loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper left')
plt.show()

model.evaluate(test_x,test_y)

test_no=int(input('please input photo number of test data: '))
plt.imshow(test_x[test_no].reshape(28,28))
print(f'it is {np.argmax(test_y[test_no])}')
pred = np.argmax(model.predict(test_x[test_no:test_no+1]))
#pred=model.predict_classes(test_x[test_no:test_no+1]) #model.predict_classes 將在2021年刪除
print(f'predict it is {pred}')

