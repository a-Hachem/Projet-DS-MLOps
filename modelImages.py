# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import os
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve
import itertools as it
from tensorflow import keras

from transformers import *
import tensorflow as tf
# import tensorflow_hub as hub
import tensorflow.keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import  Sequential

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16

from sklearn import cluster, metrics
# %matplotlib inline

# from google.colab import drive
# drive.mount('/content/drive')



# =================================================================
# ===> Required directories

current_dir= os.getcwd()

path = current_dir+"/Images/"


# =================================================================


# data = pd.read_csv('/content/drive/My Drive/Flipkart/flipkart_com-ecommerce_sample_1050.csv')

# data = pd.read_csv(current_dir+"/data_photos_0.csv", index_col=None)

data = pd.read_csv(current_dir+'/flipkart_com-ecommerce_sample_1050.csv')

data.head(3)

data.shape

data.info()

# data.uniq_id.duplicated().sum()

"""#I. Définition des catégories"""

data_cat=data.copy()
def get_product_category(category_tree):
    if 'Watches' in category_tree:
        return 'Watches'
    elif 'Home Furnishing' in category_tree:
        return 'Home Furnishing'
    elif 'Baby Care' in category_tree:
        return 'Baby Care'
    elif 'Home Decor & Festive Needs' in category_tree:
        return 'Home Decor & Festive Needs'
    elif 'Kitchen & Dining' in category_tree:
        return 'Kitchen & Dining'
    elif 'Beauty and Personal Care' in category_tree:
        return 'Beauty and Personal Care'
    elif 'Computers' in category_tree:
        return 'Computers'
    else:
        return 'Other'

data_cat['product_category'] = data['product_category_tree'].apply(get_product_category)

data_cat.columns

list_labels = data_cat['product_category'].unique().tolist()
list_labels

from sklearn.preprocessing import LabelEncoder
data_cat_num=data.copy()
label_encoder = LabelEncoder()
data_cat_num['label'] = label_encoder.fit_transform(data_cat['product_category'])

data_cat_num['product_category']=data_cat['product_category']

# =================================================================
# =================================================================

"""###II. Extraction des features image

"""

# from sklearn import preprocessing
# from google.colab import drive

# Chemin du répertoire des images
# path = '/content/drive/My Drive/Flipkart/Images'
# drive.mount("/content/drive", force_remount=True)

list_photos = [file for file in listdir(path)]
# print(len(list_photos))
# print(list_photos)

from sklearn import preprocessing

data_photos = pd.DataFrame()
data_photos[['Image', 'Label_name', 'Label']] = data_cat_num[['image', 'product_category', 'label']]
# data_photos

data_grp= data_photos.groupby(["Label_name","Label"]).count()
# data_grp

list_labels_num = data_photos['Label'].unique().tolist()
# list_labels_num

for i, label in zip(list_labels_num, list_labels):
    globals()["data_photos_train_" + str(i)] = data_photos.loc[data_photos['Label_name'] == label][:100]
    globals()["data_photos_test_" + str(i)] = data_photos.loc[data_photos['Label_name'] == label][100:150]
# data_photos_test_1

data_photos_test = pd.concat([data_photos_test_0, data_photos_test_1, data_photos_test_2, data_photos_test_3, data_photos_test_4, data_photos_test_5, data_photos_test_6])
# data_photos_test

data_photos_train = pd.concat([data_photos_train_0, data_photos_train_1, data_photos_train_2, data_photos_train_3, data_photos_train_4, data_photos_train_5, data_photos_train_6])
# data_photos_train

import os
import shutil

# Chemin du répertoire data_train
# data_train_path = '/content/drive/My Drive/Flipkart/data_train'
# data_test_path = '/content/drive/My Drive/Flipkart/data_test'

data_train_path = current_dir+"/data_train/"
data_test_path  = current_dir+"/data_test/"

# Création du répertoire data_train et son répertoire parent si nécessaire
os.makedirs(data_train_path, exist_ok=True)
os.makedirs(data_test_path, exist_ok=True)

# ===> Dataset avec augmentation des données

batch_size = 32

def dataset_fct(path, validation_split=0, data_type=None) :
    dataset = tf.keras.utils.image_dataset_from_directory(
                    path, labels='inferred', label_mode='categorical',
                    class_names=None, batch_size=batch_size, image_size=(224, 224), shuffle=True, seed=42,
                    validation_split=validation_split, subset=data_type
                    )
    return dataset

# 

dataset_train = dataset_fct(data_train_path, validation_split=0.25, data_type='training')
dataset_val = dataset_fct(data_train_path, validation_split=0.25, data_type='validation')
dataset_test = dataset_fct(data_test_path, validation_split=0, data_type=None)

# ===> Rescale data

def resize_and_rescale(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = (image / 255.0)
    return image, label

# ===> create model

def create_model_fct2() :
    # Data augmentation
    data_augmentation = Sequential([
        RandomFlip("horizontal", input_shape=(224, 224, 3)),
        RandomRotation(0.1),
        RandomZoom(0.1),
        # Rescaling(1./127.5, offset=-1.0)
      ])

    # Récupération modèle pré-entraîné
    model_base = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    for layer in model_base.layers:
        layer.trainable = False

    # Définition du nouveau modèle
    model = Sequential([
                data_augmentation,
                Rescaling(1./127.5, offset=-1),
                model_base,
                GlobalAveragePooling2D(),
                Dense(256, activation='relu'),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(7, activation='softmax')
                ])

    # compilation du modèle
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

    print(model.summary())

    return model

# =================================================================
# ===> 
# Création du modèle
with tf.device('/gpu:0'):
    model1 = create_model_fct2()


# Définir le chemin pour sauvegarder les meilleurs poids

model_path= current_dir+"/modelSaved"
os.makedirs(model_path, exist_ok=True)
model1_save_path = model_path+"/model1_best_weights.h5"

# Configuration des callbacks
checkpoint = ModelCheckpoint(model1_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# Ajouter les callbacks à la liste
callbacks_list = [checkpoint, es]

# =================================================================
# ===> Model training

n_epochs = 15

with tf.device('/gpu:0'):
    history1 = model1.fit(dataset_train,
                    validation_data=dataset_val,
                    batch_size=batch_size, epochs=n_epochs, callbacks=callbacks_list, verbose=1)



