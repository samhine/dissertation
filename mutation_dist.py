import pandas as pd
import numpy as np
from plotnine import *
import pickle

f = open('store1000_30indiv.pckl', 'rb')
data = pickle.load(f)
f.close()


indivN = 30
max_l = 0
sizes = [0 for i in range(100+1)]
#Reshaping data
for mat in range(len(data)):
    #Flatten matrices to 1D
    data[mat] = np.array(data[mat]).flatten()
    #Finding maximum length of element
    sizes[(data[mat].size)//indivN]+=1

df = pd.DataFrame(columns=["n", "count"])

df["n"] = range(0, 100+1)
df["count"]=sizes

p = ggplot(df, aes(x="n", y="count"))+geom_col()+xlab("Number of mutations")+ylab("Frequency")

p.save(filename="plots/count_dist_30indiv.png", height=5, width=10, units = 'in', dpi=1000)
