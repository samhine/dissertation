#Dataframe and vector logic modules
import pandas as pd
import numpy as np
#Machine learning modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
#Graphing module
from plotnine import *

#Contains the set of all 8 combinations - i.e. binary for numbers [0,7]
ex_set = [list(map(int, list(bin(n)[2:].zfill(3)))) for n in range(8)]

#Our training/testing sample size
sample_size = 256

#Our training/testing data
dataset = pd.DataFrame(columns=["in_1", "in_2", "in_3"])

c=0 #Counter variable for inputting into dataframe

#Adding 32 datapoints of each combination into our training data
for i in ex_set:
    for j in range(sample_size//8):
        dataset.loc[c] = i
        c+=1

#Adding gaussian noise to input/output
for i in range(3):
    mu, sigma = 0, 0.25
    noise = np.random.normal(mu, sigma, sample_size)
    dataset.iloc[:,i] += noise

#Splitting into training/testing data using sklearn (note that the x and y data are pulled from the same dataset)
X_train, X_test, y_train, y_test = train_test_split(dataset, dataset, test_size=0.5)

model = Sequential([
    Dense(2, input_dim=3),
    Dense(3, activation=tf.keras.activations.sigmoid)
    ])

sgd = optimizers.SGD(lr=0.1)
model.compile(optimizer = sgd,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

result = model.fit(X_train, y_train,
                   validation_data=(X_test, y_test),
                   epochs = 100)

print(model.summary())

for ex in ex_set:
    test_in = np.array(ex,).reshape(1,3)
    test_out = model.predict(test_in)

    print(test_in)
    print(test_out)
