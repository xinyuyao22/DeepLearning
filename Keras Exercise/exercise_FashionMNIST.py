# %% -------------------------------------------------------------------------------------------------------------------
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from keras.initializers import glorot_uniform
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import h5py
from sklearn.metrics import confusion_matrix
# % -------------------------------------------------------------------------------------
# Fit a MLP to the FashionMNIST dataset: https://github.com/zalandoresearch/fashion-mnist
# % -------------------------------------------------------------------------------------

LR = 1e-3
N_NEURONS = (100, 200, 100)
N_EPOCHS = 20
BATCH_SIZE = 512
dropout = 0.2

# 1. Download the data using keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# Reshapes to (n_examples, n_pixels), i.e, each pixel will be an input feature to the model
x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
x_train, x_test = x_train/255, x_test/255
y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)

# 2. Try using more/less layers and different hidden sizes to get a good better fit. Also play with the dropout.
model = Sequential([  # The dropout is placed right after the outputs of the hidden layers.
    Dense(N_NEURONS[0], input_dim=784, kernel_initializer=glorot_uniform(0)),  # This sets some of these outputs to 0, so that
    Activation("relu"),  # a random dropout % of the hidden neurons are not used during each training step,
    Dropout(dropout),  # nor are they updated. The Batch Normalization normalizes the outputs from the hidden
    BatchNormalization()  # activation functions. This helps with neuron imbalance and can speed training significantly.
])  # Note this is an actual layer with some learnable parameters. It's not just min-maxing or standardizing.
# Loops over the hidden dims to add more layers
for n_neurons in N_NEURONS[1:]:
    model.add(Dense(n_neurons, activation="relu", kernel_initializer=glorot_uniform(0)))
    model.add(Dropout(dropout, seed=0))
    model.add(BatchNormalization())
# Adds a final output layer with softmax to map to the 10 classes
model.add(Dense(10, activation="softmax", kernel_initializer=glorot_uniform(0)))
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])
filepath = "mlp_xinyu.hdf5"
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test),
          callbacks=[ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True),
                                             EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0)])
# 3. Add an option to save the model on each epoch, and stop saving it when the validation
# loss begins to increase (early stopping) - https://keras.io/callbacks/: ModelCheckpoint


# 4. Add an option to only test the model, by loading the model you saved on the training phase

model.load_weights(filepath)

print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")

# 5. Print out the confusion matrix
y_pred = np.argmax(model.predict(x_test), axis=1)
y_test = np.argmax(y_test, axis=1)
print(confusion_matrix(y_pred, y_test))

labels= {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}

# 6. Define a function to show some images that were incorrectly classified

def mis_class(y_pred, y_test):
    misclassified = np.where(y_test != y_pred)
    for i in misclassified[0][:5]:
        image = x_test[i, :]
        pixels = image.reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.title(labels[y_pred[i]])
        plt.show()
    return

mis_class(y_pred, y_test)

# %% -------------------------------------------------------------------------------------------------------------------
