# %% --------------------------------------- Imports -------------------------------------------------------------------

import numpy as np
from keras.models import load_model

import os
os.system("sudo pip install PIL")
from PIL import Image

def predict(images):
    # On the actual exam it will be a list of paths.
    # %% --------------------------------------------- Data Prep -------------------------------------------------------
    # Write any data prep you used during training
    def Reformat_Image(ImageFilePath, widest, highest, flip=False, correction=False):
        image = Image.open(ImageFilePath, 'r')
        image.thumbnail((widest, highest))
        image_size = image.size
        width = image_size[0]
        height = image_size[1]
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if correction:
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
        pix = np.array(background.getdata()).reshape((widest, highest, 3))
        image.close()
        return pix
    x_test = []
    for im_path in images:
        x_test.append(Reformat_Image(im_path, 128, 128))
    x_test = np.asarray(x_test).reshape((-1, 128*128*3))
    model1 = load_model('mlp_xinyu_yao1.hdf5')
    y_pred1 = model1.predict(x_test)
    x_test = []
    for im_path in images:
        x_test.append(Reformat_Image(im_path, 128, 128, flip=True))
    x_test = np.asarray(x_test).reshape((-1, 128*128*3))
    model2 = load_model('mlp_xinyu_yao2.hdf5')
    y_pred2 = model2.predict(x_test)
    x_test = []
    for im_path in images:
        x_test.append(Reformat_Image(im_path, 128, 128, flip=True, correction=True))
    x_test = np.asarray(x_test).reshape((-1, 128*128*3))
    model3 = load_model('mlp_xinyu_yao3.hdf5')
    y_pred3 = model3.predict(x_test)
    y_pred = np.argmax(np.add(np.add(y_pred1, y_pred2), y_pred3), axis=1)
    return y_pred, model1, model2, model3
    # If using more than one model to get y_pred, do the following:
    # return y_pred, model1, model2  # If you used two models
    # return y_pred, model1, model2, model3  # If you used three models, etc.
