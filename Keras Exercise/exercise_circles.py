# %% -------------------------------------------------------------------------------------------------------------------
import math
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.initializers import glorot_uniform
import matplotlib.pyplot as plt
# ---------------------------------------------
# Learn a circular Decision Boundary using  MLP
# ---------------------------------------------

LR = 0.01
N_NEURONS = 10
N_EPOCHS = 300

# 1. Define a function to generate the y-points for a circle, taking as input the x-points and the radius r.

def fun(x, r):
    y = [[math.sqrt(r**2 - x[i]**2), -math.sqrt(r**2 - x[i]**2)] for i in range(len(x))]
    y = np.ravel(np.asarray(y))
    return y

# 2. Use this function to generate the data to train the network. Label points with r=2 as 0 and points with r=4 as 1.
# Note that for each value on the x-axis there should be two values on the y-axis, and vice versa.

x0 = np.linspace(-2, 2, 400)
y0 = fun(x0, 2)
x1 = np.linspace(-4, 4, 400)
y1 = fun(x1, 4)
x00 = np.repeat(x0, 2)
x11 = np.repeat(x1, 2)
x = np.vstack((np.hstack((x00.T, x11.T)), np.hstack((y0, y1)))).T
y = np.asarray([0] * len(x00) + [1] * len(x11)).reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2, stratify=y)
y_train_cat, y_test_cat = to_categorical(y_train, num_classes=2), to_categorical(y_test, num_classes=2)

# 3. Choose the right number of input and output neurons, define and train a MLP to classify this data.

model = Sequential([
    Dense(N_NEURONS, input_dim=2, activation= 'sigmoid',kernel_initializer=glorot_uniform(42)),
    BatchNormalization(),
    Dense(N_NEURONS, activation= 'sigmoid',kernel_initializer=glorot_uniform(42)),
    BatchNormalization(),
    Dense(2, kernel_initializer=glorot_uniform(42)),  # Output layer with softmax to map to the two classes
    Activation("softmax")
])
# Compiles using categorical cross-entropy performance index and tracks the accuracy during training
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train_cat, batch_size=len(x_train), epochs=N_EPOCHS)

# 4. Use model.evaluate to get the final accuracy on the whole set and print it out

print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test_cat)[1], "%")

# 5. Make a contour plot of the MLP as a function of the x and y axis. You can follow
# https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_versus_svm_iris.html

# create a mesh to plot in
xx, yy = np.meshgrid(x1, x1)

color_map = {0: (0, 0, .9), 1: (1, 0, 0)}

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.argmax(Z, axis=1).reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
colors = [color_map[y] for y in np.ravel(y_train).tolist()]
plt.scatter(x_train[:, 0], x_train[:, 1], c=colors, edgecolors='black')
plt.show()

# %% -------------------------------------------------------------------------------------------------------------------
