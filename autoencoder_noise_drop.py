import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import random
import pickle
from plotnine import *

#from diss import genmat

#Define 6x9 matrix
#Input Shape - (1, 6, 9)
#x=inputs, y,z=input shape

##inpt = []
##for x in range(0,10000):
##    mat = []
##    for y in range(0,6):
##        row = []
##        for z in range(0,9):
##            row.append(random.randint(0,1))
##        row=np.array(row)
##        mat.append(row)
##    mat = np.array(mat).flatten()
##    inpt.append(mat)

#File input
f = open('store100000_8x9.pckl', 'rb')
inpt = pickle.load(f)
f.close()

#Reorganising data for model input
for mat in range(len(inpt)):
    inpt[mat] = np.array(inpt[mat]).flatten()
inpt = np.array(inpt)
inpt = inpt.reshape(inpt.shape[0], inpt.shape[1])

f = open('store100000_8x9_unknoise.pckl', 'rb')
inpt_noise = pickle.load(f)
f.close()

#Reorganising data for model input (for noise)
for mat in range(len(inpt_noise)):
    inpt_noise[mat] = np.array(inpt_noise[mat]).flatten()
inpt_noise = np.array(inpt_noise)
inpt_noise = inpt_noise.reshape(inpt_noise.shape[0], inpt_noise.shape[1])

X_train, X_test, y_train, y_test = train_test_split(inpt_noise, inpt, test_size=0.2, random_state=123)

#Number of neuron in input
neuron_n = 72

#Constructing model
model = Sequential([
    Dense(120, input_dim=neuron_n),
    Dropout(0.7),
    Dense(neuron_n, activation=tf.keras.activations.sigmoid)
    ])

optim = optimizers.RMSprop(lr=0.005)
model.compile(optimizer = optim,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

print(model.summary())

result = model.fit(X_train, y_train,
                   validation_data=(X_test, y_test),
                   epochs = 5)

#Get final output
layer_name = 'dense_1' #Selects last layer

final_layer_model = Model(inputs=model.input,
                          outputs=model.get_layer(layer_name).output)

final_output = final_layer_model.predict(X_test)

sample_N = 30

print("input mat \n", X_test[sample_N])
print("output mat \n", final_output[sample_N])

df = pd.DataFrame(columns=["indiv", "mut_pos", "in_val", "out_val", "ideal_out_val"])

c = 0
mutN = 9
indivN = 8
for indiv in range(0,indivN):
    for mut in range(0,mutN):
        df.loc[c] = [indiv, mut, X_test[sample_N][indiv*mutN+mut], final_output[sample_N][indiv*mutN+mut], y_test[sample_N][indiv*mutN+mut]]
        c+=1

p = ggplot(df, aes("factor(mut_pos)", "factor(indiv)", fill="in_val")) + geom_tile(aes(width=.95, height=.95)) + scale_fill_gradient(low="#000000", high="#ffffff", limits=[0,1]) + xlab("Mutation positions") + ylab("Individuals")
p.save(filename = 'plots/test/8x9_heat_in_model6_1epoch_noise.png', height=5, width=10, units = 'in', res=1000)

p2 = ggplot(df, aes("factor(mut_pos)", "factor(indiv)", fill="out_val")) + geom_tile(aes(width=.95, height=.95)) + scale_fill_gradient(low="#000000", high="#ffffff", limits=[0,1]) + xlab("Mutation positions") + ylab("Individuals")
p2.save(filename = 'plots/test/8x9_heat_out_model6_1epoch_noise.png', height=5, width=10, units = 'in', res=1000)

p2 = ggplot(df, aes("factor(mut_pos)", "factor(indiv)", fill="ideal_out_val")) + geom_tile(aes(width=.95, height=.95)) + scale_fill_gradient(low="#000000", high="#ffffff", limits=[0,1]) + xlab("Mutation positions") + ylab("Individuals")
p2.save(filename = 'plots/test/8x9_heat_out_ideal_model6_1epoch_noise.png', height=5, width=10, units = 'in', res=1000)

#Get intermediate output
layer_name = 'dense' #Name of intermediate layer

intermediate_layer_model = Model(inputs=model.input,
                          outputs=model.get_layer(layer_name).output)

intermediate_output = intermediate_layer_model.predict(X_test)

sample_N = 30

print("input mat \n", X_test[sample_N])
print("output mat \n", intermediate_output[sample_N])

df = pd.DataFrame(columns=["neuron_n", "val", "idn"]) #idn is a column containing 1's for x axis

c = 0
neuronN = 20 #Number of neurons in this layer
for i in range(0,10):
    for j in range(0,12):
        df.loc[c] = [i, intermediate_output[sample_N][i*j+j], j]
        c+=1
print(df)
p = ggplot(df, aes("factor(idn)", "factor(neuron_n)", fill="val")) + geom_tile(aes(width=.95, height=.95)) + scale_fill_gradient(low="#000000", high="#ffffff", limits=[0,1]) + xlab("") + ylab("Neurons")

p.save(filename = 'plots/test/8x9_heat_inter_model6_1epoch_noise.png', height=5, width=3.33, units = 'in', res=1000)
