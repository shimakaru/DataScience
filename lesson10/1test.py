# Scikit-Learn ≥0.20 is required
import sklearn
# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
np.random.seed(42)
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd





mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ann"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
#訓練セットを「訓練セットと検証セット」に分ける。ついでにxの値を0〜１にする。
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.
#print(y_train)
#plt.imshow(X_train[-2], cmap="binary")
#plt.axis('off')
#plt.show()
#yのインデックスの意味がこれらしい。別になくてもいい気はするが一応書いとこう
class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
#-----------------学習モデル------------------
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()
#モデルの可視化
#keras.utils.plot_model(model, "my_fashion_mnist_model.png", show_shapes=True)
model.compile(loss="sparse_categorical_crossentropy",
	optimizer="sgd",
	metrics=["accuracy"])
#------------------学習-----------------------
history = model.fit(X_train, y_train, epochs=30,validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
save_fig("keras_learning_curves_plot")
plt.show()


