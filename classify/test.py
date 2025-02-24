import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.models import load_model

# Load data
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.utils.vis_utils import plot_model


# Split data into training and testing sets
data1 = np.loadtxt('ce6_0.txt')
data1 = data1[:, 1:]
y1 = [1 for i in range(data1.shape[0])]
# print(data1.shape[0]) #301
data2 = np.loadtxt('Ru_0.txt')
data2 = data2[:, 1:]
y2 = [2 for i in range(data2.shape[0])]
# print(data2.shape[0]) #326

data3 = np.loadtxt('hpts_5.txt')
data3 = data3[:, 1:]
y3 = [3 for i in range(data3.shape[0])]
# print(data3.shape[0]) #326
merged_y = np.concatenate((y1, y2, y3), axis=0)
merged_data = np.concatenate((data1, data2, data3), axis=0)
# print(type(merged_y), type(merged_data))
X = merged_data
print(X.shape)
y = merged_y
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_train.shape[0])
# Reshape data
X_train = X_train.reshape(X_train.shape[0], 1, 20, 1)
X_test = X_test.reshape(X_test.shape[0], 1, 20, 1)

# Convert labels to categorical data
y_train = to_categorical(y_train)
y_train = y_train[:, :4]
print("y_train is ", y_train)
y_test = to_categorical(y_test)
y_test = y_test[:, :4]
print("y_test is ", y_test)

# Define model
model = Sequential()
model.add(Conv2D(32, kernel_size=(1, 3), activation='relu', input_shape=(1, 20, 1)))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))
# model = Sequential([
#     Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
#     Dense(64, activation='relu'),
#     Dense(3, activation='softmax')  # 输出层使用softmax激活函数
# ])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5)
print(X_train.shape)
print(y_train.shape)
# Train model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=16,
                    callbacks=[checkpoint, early_stop])
#

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# Load best model
model = load_model('model.h5')

# Evaluate model
score = model.evaluate(X_test, y_test, verbose=0)
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

test_data = X_test[0]
print(X_test[0])

# Use the trained model to predict the class of the test data point
prediction = model.predict(test_data.reshape(1, 1, 20, 1))

# Print the predicted class
print(prediction.argmax())
# Plot model architecture
plot_model(model, to_file='model.png', show_shapes=True)
