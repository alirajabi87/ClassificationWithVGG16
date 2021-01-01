"""
This program uses VGG16 as a pretrained model in combination with some new Dense layers
to create a prediction model for small database created from a large one
"""

import os, ctypes, sys, subprocess

# This code prevent INFO, WARNING, and ERROR messages to be printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
"""
 This section is for the admin privilege to run the LimitedDatabase file and
 create symbolic links
"""


def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def create():
    if is_admin():
        subprocess.run(args=args, shell=True)
        print("The New database has been successfully created")
    else:
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)

"""
    The database is: 
        Fruits 360 dataset: A dataset of images containing fruits and vegetables
        Version: 2020.05.18.0
    It can be downloaded from: https://www.kaggle.com/moltean/fruits
"""
original_path = './fruits-360/'
dst_path = './DATA/smallDatabase'
NumberofClasses = 10
args = f'python LimitedDatabase.py {original_path} {dst_path} {NumberofClasses}'
# #### To create Sub-DataSet uncomment the following command
# create()
###############################################################################
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools

folders = glob('./DATA/smallDatabase/Training/*')
print(len(folders))

Training_Images = glob(dst_path + '/Training/*/*.jp*g')
Validation_Images = glob(dst_path + '/Validation/*/*.jp*g')

"""
Checking the images
"""
imgPath = np.random.choice(Training_Images)
img = load_img(imgPath)
img = np.array(img)
print(f"The min of image: {img.min()}, the Max of image is: {img.max()}")
plt.title(imgPath.split('\\')[-2])
plt.imshow(img)
plt.show()
###########################################################
# Global parameters
ImageSize = (150, 150, 3)
batchSize = 32

"""
Generating Images from folders
"""
generatorImage = ImageDataGenerator(rotation_range=30,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    height_shift_range=0.1,
                                    width_shift_range=0.1,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    rescale=1. / 255,
                                    preprocessing_function=preprocess_input)

path_train = dst_path + '/Training'
path_valid = dst_path + '/Validation'

Train_gen = generatorImage.flow_from_directory(path_train,
                                               target_size=ImageSize[:2],
                                               shuffle=True,
                                               class_mode='categorical',
                                               batch_size=batchSize,
                                               interpolation='nearest')
valid_gen = generatorImage.flow_from_directory(path_valid,
                                               target_size=ImageSize[:2],
                                               shuffle=True,
                                               class_mode='categorical',
                                               batch_size=batchSize,
                                               interpolation='nearest')

Steps_per_Train = Train_gen.n // batchSize
Steps_per_Valid = valid_gen.n // batchSize

"""
Checking the first image in train database (if you comment rescale=1. / 255
the size will be different
"""
for x, y in Train_gen:
    print(x[0].max(), x[0].min())
    break

"""
Creating the Model based on VGG16
"""

vgg = VGG16(input_shape=ImageSize, weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)
prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint

path_checkpoint = './tmp'
if not os.path.exists(path_checkpoint):
    os.mkdir(path_checkpoint)

model_checkpoint_callback = ModelCheckpoint(monitor='val_accuracy',
                                            filepath=path_checkpoint,
                                            save_weights_only=True,
                                            mode='max',
                                            save_best_only='True')
epochs = 10
res = model.fit(Train_gen,
                epochs=epochs,
                validation_data=valid_gen,
                steps_per_epoch=Steps_per_Train,
                validation_steps=Steps_per_Valid,
                callbacks=[model_checkpoint_callback],
                verbose=2)
model.save(f'ModelFruits_{len(folders)}.h5')

plt.plot(res.history['accuracy'])
plt.show()

results = pd.DataFrame(model.history.history)
results[['loss', 'val_loss']].plot()
plt.show()


"""
For loading and evaluating the model please uncomment the following commands
"""
# model = load_model('ModelFruits_10.h5')

print(model.evaluate(valid_gen, verbose=2))


"""
Defining the functions to obtain the confusion matrix and plot it
"""

def get_Confusion_matrix(data, N):
    print(f"Generating confusion matrix for {N} classes")
    y_pred = model.predict(data)
    predictions = y_pred > 0.7
    pred2 = np.argmax(y_pred, axis=1)
    y_true = data.classes
    cm = confusion_matrix(data.classes, pred2)
    return (cm, pred2, y_true)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()


# Create the labels for the classes
labels = []
for k, v in Train_gen.class_indices.items():
    labels.append(k)


# pred = model.predict(valid_gen)
# predictions = np.argmax(pred, axis=1)
# print(len(valid_gen.classes), len(predictions))
# print(classification_report(valid_gen.classes, predictions))

"""
    Creating and plotting the confusion matrix, prediction classes and True classes
"""
valid_cm, y_pred_valid, y_true_valid = get_Confusion_matrix(valid_gen, len(folders))
# train_cm, y_pred_train, y_true_train = get_Confusion_matrix(Train_gen, len(folders))

plot_confusion_matrix(valid_cm, title="For Validation", classes=labels)
# plot_confusion_matrix(train_cm, title="For Training", classes=labels)

"""
    creating the classification report
"""
# print('############################## Training #############################')
# print(classification_report(y_true=y_true_train, y_pred=y_pred_train, target_names=labels))
print('############################## Validation #############################')
print(classification_report(y_true=y_true_valid, y_pred=y_pred_valid, target_names=labels))
