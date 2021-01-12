from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

input_size = 784
hidden_neurons = 200
classes = 10
input_shape = (28, 28, 1)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
x_train = X_train.reshape(60000,  28, 28, 1)/255
x_test = X_test.reshape(10000, 28, 28, 1)/255
y_train = to_categorical(Y_train, 10)
y_test = to_categorical(Y_test)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')
History = model.fit(x_train, y_train, validation_split=0.2,
                    epochs=10, batch_size=1024)

score = model.evaluate(x_train, y_train)
print('train accuracy:', score[1])
print('train loss:', score[0])
score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])
print('Test loss:', score[0])

print(History.history)

model.save('model')
