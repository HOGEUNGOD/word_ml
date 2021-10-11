import glob
import sys
import os
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
from tensorflow import keras
from functools import partial
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def data_load():
    files = glob.glob('./data/*')
    if not files:
        print("Chck Phath")
        sys.exit()
    else:
        files.remove(r'./data\raw_data')
        files.remove(r'./data\raw_data_')

    _feature=[]
    label=[]
    for address in files:
        label_ = os.path.basename(address)
        for num in glob.glob(address+'/*'):
            _feature.extend(np.load(num))
            label.extend([label_]*len(np.load(num)))
        print(label_)
    switch = 0

    for i in _feature:
        if switch == 0:
            feature = np.array([i])
            switch = 1+ switch

        else:
            feature = np.vstack((feature, [i]))
    target = np.array(label).flatten()
    return feature, target.reshape((-1,))

features, label = data_load()
encoder = LabelEncoder()
target = encoder.fit_transform(label)


X_train,  X_valid, y_train, y_valid = train_test_split(features, target, test_size=0.15, random_state=0)
X_train = X_train / 255.0
X_valid = X_valid / 255.0

y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
number_of_classes = y_valid.shape[1]


DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")
model = keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=3, input_shape=[210, 210, 5]),
    DefaultConv2D(filters=64),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=512),
    DefaultConv2D(filters=512),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=number_of_classes, activation="softmax"),
])

model.compile(loss="categorical_crossentropy", optimizer='Rmsprop', metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=300, validation_data=(X_valid, y_valid))
score = model.evaluate(X_valid, y_valid)

training_loss = history.history["loss"]
test_loss = history.history["val_loss"]
epoch_count = range(1, len(training_loss)+1)

training_acc = history.history["accuracy"]
test_acc = history.history["val_accuracy"]

plt.plot(epoch_count, training_loss, "r-")
plt.plot(epoch_count, test_loss, "b-")
plt.legend(["training_loss", "test_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim(0,0.1)

#%%

plt.plot(epoch_count, training_loss, "r-")
plt.plot(epoch_count, test_loss, "b-")
plt.legend(["training_loss", "test_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('./figure/loss.png',dpi=300,bbox_inches='tight')

#%%

plt.plot(epoch_count, training_acc, "r-")
plt.plot(epoch_count, test_acc, "b-")
plt.legend(["training_acc", "test_acc"])
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.savefig('./figure/acc.png',dpi=300,bbox_inches='tight')

#%%

for i in features:

    fig, ax = plt.subplots(1,5)
    ax[0].imshow(i[:,:,0], cmap='magma')
    ax[1].imshow(i[:,:,1], cmap='magma')
    ax[2].imshow(i[:,:,2], cmap='magma')
    ax[3].imshow(i[:,:,3], cmap='magma')
    ax[4].imshow(i[:,:,4], cmap='magma')
    plt.show()
    plt.close()

#%%

plt.plot()
plt.imshow(features[0][:,:,0], cmap='magma')
