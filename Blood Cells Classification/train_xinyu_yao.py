import os
from os import listdir

import numpy as np
import optuna
from PIL import Image
from keras.initializers import glorot_uniform
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import cohen_kappa_score

os.chdir("/home/ubuntu/train/")
mypath = "/home/ubuntu/train/"
images = [f for f in listdir(mypath) if f.endswith(".png")]
images.sort()
txts = [f for f in listdir(mypath) if f.endswith(".txt")]
txts.sort()

y = []
for f in txts:
    with open(f, 'r') as file:
        y.append(file.read())
txts = None

index0 = [i for i, x in enumerate(y) if x =='red blood cell']
index1 = [i for i, x in enumerate(y) if x =='ring']
index2 = [i for i, x in enumerate(y) if x == 'schizont']
index3 = [i for i, x in enumerate(y) if x =='trophozoite']

image_list = []
for i in index0:
    image_list.append(images[i])

def rotate_image(images, index, max):
    image_list = []
    angle = 360 /(max/len(index))
    for i in index:
        image = Image.open(images[i])
        for j in range(int(max/len(index))):
            path = str(j) + images[i]
            rotated = image.rotate(angle * (j + 1))
            rotated.save(path)
            image_list.append(path)
        image.close()
    return image_list

image_list = image_list + rotate_image(images, index1, len(index0))
image_list = image_list + rotate_image(images, index2, len(index0))
image_list = image_list + rotate_image(images, index3, len(index0))

images = None

y = [0] * len(index0) + [1] * int(len(index0)/len(index1)) * len(index1) + [2] * int(len(index0)/len(index2)) * len(index2) + [3] * int(len(index0)/len(index3)) * len(index3)
index3 = None
index2 = None
index1 = None
index0 = None

def Reformat_Image(image_path, widest, highest):
    image = Image.open(image_path)
    image.thumbnail((widest, highest))
    image_size = image.size
    width = image_size[0]
    height = image_size[1]
    gamma = 1.2
    result_img = Image.new("RGB", (width, height))
    dict = {}
    for i in range(256):
        value = round(pow(i / 255, (1 / gamma)) * 255, 0)
        if value >= 255:
            value = 255
        dict[i] = int(value)
    for x in range(width):
        for y in range(height):
            value1 = dict[image.getpixel((x, y))[0]]
            value2 = dict[image.getpixel((x, y))[1]]
            value3 = dict[image.getpixel((x, y))[2]]
            result_img.putpixel((x, y), (int(value1), int(value2), int(value3)))
    background = Image.new('RGB', (widest, highest), (255, 255, 255))
    offset = (int(round(((widest - width) / 2), 0)), int(round(((highest - height) / 2), 0)))
    background.paste(image, offset)
    pix = np.ravel(np.array(background.getdata()))
    image.close()
    os.remove(image_path)
    return pix

x = []
for im in image_list:
    x.append(Reformat_Image(im, 128, 128))

image_list = None

input_dim=128*128*3
x = np.asarray(x).reshape((-1, input_dim))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2, stratify=y)
x = None
y = None
y_train, y_test = to_categorical(y_train, num_classes=4), to_categorical(y_test, num_classes=4)

study = optuna.create_study(study_name = 'cell', storage = 'sqlite:///cell.db', load_if_exists = True, direction='minimize')
filepath = "mlp_xinyu_yao.hdf5"
def objective(trial):
    global X_train_std, y_train, y_test
    model = Sequential([
        Dense(trial.suggest_int('neurons', 200, 500), input_dim=128*128*3, activation='relu', kernel_initializer=glorot_uniform(0)),
        Dropout(trial.suggest_uniform('dropout', 0.2, 0.5), seed=0),
        BatchNormalization()
    ])
    n_neurons = trial.suggest_int('n_neurons', 3, 7)
    for i in range(2, n_neurons):
        model.add(Dense(trial.suggest_int('neurons', 200, 500), activation='relu', kernel_initializer=glorot_uniform(0)))
        model.add(Dropout(trial.suggest_uniform('dropout', 0.2, 0.5), seed=0))
        model.add(BatchNormalization())
    model.add(Dense(4, activation='softmax', kernel_initializer=glorot_uniform(0)))
    model.compile(optimizer=Adam(lr=trial.suggest_loguniform('lr', 1e-4, 0.1)), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=trial.suggest_int('batch_size', 512, 10000), shuffle=True,
              epochs=200, validation_data=(x_test, y_test), callbacks= [ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True),
                                                                        EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0)])
    return -cohen_kappa_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1))

study.optimize(objective, n_trials = 100, timeout=1800)
print('Number of finished trials: {}'.format(len(study.trials)))
print('Best trial:')
trial = study.best_trial
print('Values:{}'.format(trial.value))
print('Params:')
for key, value in trial.params.items():
    print('{}:{}'.format(key, value))