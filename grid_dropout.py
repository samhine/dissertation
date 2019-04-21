import pandas as pd
import numpy as np
from plotnine import *
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from operator import mul
import random
import pickle
from math import log

#Our grid only cares for extrapolation layers (i.e. those above the magnitude of input)
grid = { "hidden_layers" : [(72,), (90,), (120,)],
         "lr" : [0.001, 0.005, 0.01, 0.05, 0.1],
         "dropout_frac" : [0.1,0.3,0.5,0.7,0.9]
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

df = pd.DataFrame(columns = ["shape", "optim", "lr", "val_loss", "val_acc", "frac"])

totl = [len(grid[key]) for key in grid]
tot = 1
for n in totl:
    tot*=n

c=0
performances = []
print("\n>> There are "+str(tot)+" configurations to check")
for shape in grid["hidden_layers"]:
    for lr in grid["lr"]:
        for frac in grid["dropout_frac"]:
            print("\n>> Model defined as ", shape, lr, frac)
            c+=1
            print(">> We are "+str((c/tot)*100)+"% done") 
            
            model = Sequential() #Redefine model on each iteration
            
            model.add(Dense(shape[0], input_dim=input_dim))
            model.add(Dropout(frac))
            if len(shape)>1:
                for layer in shape[1:]:
                    model.add(Dense(layer))
                    model.add(Dropout(frac))
            model.add(Dense(input_dim, activation=tf.keras.activations.sigmoid))


            optimiser_obj = optimizers.RMSprop(lr=lr)

            
            model.compile(loss='binary_crossentropy',
                          optimizer = optimiser_obj,
                          metrics = ["accuracy"])
            result = model.fit(X_train, y_train,
                               validation_data = (X_test, y_test),
                               epochs=5)

            perform = {"shape" : str(shape),
                       "optim" : "rmsprop",
                       "lr" : str(lr),
                       "frac" : str(frac),
                       "val_loss" : result.history["val_loss"][-1],
                       "val_acc" : result.history["val_acc"][-1]
                       }
            performances.append(perform)
            df = df.append(perform, ignore_index=True)

for run in performances:
    print(run, "\n")

print(df)

df.to_csv("grid_results_8x9_dropout_5epoch.csv", index=False)

##p = ggplot(df, aes('optim', 'val_acc', color='lr'))+geom_point()
##p.save(filename = 'plots/by_lr.png', height=5, width=5, units = 'in', dpi=1000)
##
##p2 = ggplot(df, aes('optim', 'val_acc', color='shape'))+geom_point()
##p2.save(filename = 'plots/by_shape.png', height=5, width=5, units = 'in', dpi=1000)

p3 = ggplot(df, aes('shape', 'val_acc', color='lr', shape='frac'))+geom_point(position=position_dodge(width = 0.3))+labs(title="Grid results for an AA of input size "+str(input_dim))+xlab("Hidden layer shape")+ylab("Validation accuracy")
            
p3.save(filename = 'plots/8x9_drop_acc_5epoch.png', height=5, width=10, units = 'in', dpi=1000)

p3 = ggplot(df, aes('shape', 'val_loss', color='lr', shape='frac'))+geom_point(position=position_dodge(width = 0.3))+labs(title="Grid results for an AA of input size "+str(input_dim))+xlab("Hidden layer shape")+ylab("Validation loss")
            
p3.save(filename = 'plots/8x9_drop_loss_5epoch.png', height=5, width=10, units = 'in', dpi=1000)


df['val_loss'] = df['val_loss'].apply(log)

p3 = ggplot(df, aes('shape', 'val_loss', color='lr', shape='frac'))+geom_point(position=position_dodge(width = 0.3))+labs(title="Grid results for an AA of input size "+str(input_dim))+xlab("Hidden layer shape")+ylab("Validation log loss")
            
p3.save(filename = 'plots/8x9_drop_logloss_5epoch.png', height=5, width=10, units = 'in', dpi=1000)
