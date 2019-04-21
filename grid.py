import pandas as pd
import numpy as np
from plotnine import *
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from operator import mul
import random
import pickle

grid = { "hidden_layers" : [(12,12), (20,20), (30,), (60,30,60), (48,), (60,60), (72,72)],
         "optimiser" : ['rmsprop'],
         "lr" : [0.001, 0.005, 0.01, 0.05, 0.1],
         "o_act" : ['linear', 'relu', 'sigmoid']
        }

f = open('store100000_8x9.pckl', 'rb')
data = pickle.load(f)
f.close()

#Reshaping data
for mat in range(len(data)):
    data[mat] = np.array(data[mat]).flatten()
data = np.array(data)
print(data.shape)
data = data.reshape(data.shape[0], data.shape[1])
print(data.shape)

X_train, X_test, y_train, y_test = train_test_split(data, data, test_size=0.2)
input_dim = len(data[0])

df = pd.DataFrame(columns = ["shape", "optim", "lr", "val_loss", "val_acc"])

totl = [len(grid[key]) for key in grid]
tot = 1
for n in totl:
    tot*=n

c=0
performances = []
print("\n>> There are "+str(tot)+" configurations to check")
for shape in grid["hidden_layers"]:
    for optimiser in grid["optimiser"]:
        for lr in grid["lr"]:
            for act in grid["o_act"]:
                print("\n>> Model defined as ", shape, optimiser, lr, act)
                c+=1
                print(">> We are "+str((c/tot)*100)+"% done")

                if act=="linear":
                    activation = tf.keras.activations.linear
                elif act=="relu":
                    activation = tf.keras.activations.relu
                elif act=="sigmoid":
                    activation = tf.keras.activations.sigmoid
                
                model = Sequential() #Redefine model on each iteration
                
                model.add(Dense(shape[0], input_dim=input_dim))
                if len(shape)>1:
                    for layer in shape[1:]:
                        model.add(Dense(layer))
                model.add(Dense(input_dim, activation=activation))

                #Defining optimisers
                if optimiser=="adam":
                    optimiser_obj = optimizers.Adam(lr=lr)
                elif optimiser=="sgd":
                    optimiser_obj = optimizers.SGD(lr=lr)
                elif optimiser=="rmsprop":
                    optimiser_obj = optimizers.RMSprop(lr=lr)
                elif optimiser=="adagrad":
                    optimiser_obj = optimizers.Adagrad(lr=lr)
                else:
                    print(">> Optimiser not defined in grid")
                    break
                
                model.compile(loss='binary_crossentropy',
                              optimizer = optimiser_obj,
                              metrics = ["accuracy"])
                result = model.fit(X_train, y_train,
                                   validation_data = (X_test, y_test),
                                   epochs=10)

                perform = {"shape" : str(shape),
                           "optim" : optimiser,
                           "lr" : str(lr),
                           "o_act" : act,
                           "val_loss" : result.history["val_loss"][-1],
                           "val_acc" : result.history["val_acc"][-1]
                           }
                performances.append(perform)
                df = df.append(perform, ignore_index=True)

for run in performances:
    print(run, "\n")

print(df)

df.to_csv("grid_results_8x9_10epoch.csv", index=False)

##p = ggplot(df, aes('optim', 'val_acc', color='lr'))+geom_point()
##p.save(filename = 'plots/by_lr.png', height=5, width=5, units = 'in', dpi=1000)
##
##p2 = ggplot(df, aes('optim', 'val_acc', color='shape'))+geom_point()
##p2.save(filename = 'plots/by_shape.png', height=5, width=5, units = 'in', dpi=1000)

p3 = ggplot(df, aes('shape', 'val_acc', color='lr', shape='o_act'))+geom_point(position=position_dodge(width = 0.3))+labs(title="Grid results for an AA of input size "+str(input_dim))+xlab("Hidden layer shape")+ylab("Validation accuracy")
            
p3.save(filename = 'plots/8x9_acc_10epoch.png', height=5, width=10, units = 'in', dpi=1000)

p3 = ggplot(df, aes('shape', 'val_loss', color='lr', shape='o_act'))+geom_point(position=position_dodge(width = 0.3))+labs(title="Grid results for an AA of input size "+str(input_dim))+xlab("Hidden layer shape")+ylab("Validation loss")
            
p3.save(filename = 'plots/8x9_loss_10epoch.png', height=5, width=10, units = 'in', dpi=1000)


