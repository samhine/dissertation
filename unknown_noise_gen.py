import pandas as pd
import numpy as np
import random
import pickle

f = open('store100000_8x9.pckl', 'rb')
inpt = pickle.load(f)
f.close()

#For each matrix
for mat in range(len(inpt)):
    #We have 72 values, so will explicitly flip 7 values
    for k in range(7):
        #Randomly select indices
        i, j = random.randint(0,7), random.randint(0,8)
        #Set value to 0.5 at these indices
        inpt[mat][i][j] = 0.5


f = open('store100000_8x9_unknoise.pckl', 'wb')
pickle.dump(inpt, f)
f.close()
