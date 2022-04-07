
from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

def bumpchart(df, xlabels=["SFV",250,500,750,1000],xlabel="Simulations"):
    MAX_RANK = len(df.columns)
    NUMBER_OF_ROWS = df.shape[0]
    x = range(1, df.shape[0]+1)
    # Plot columns
    for col in df.columns:
        plt.plot(x, df[col], label=col)
    
    # Axes
    plt.xlim((1, NUMBER_OF_ROWS))
    plt.xticks(np.arange(1, NUMBER_OF_ROWS+1, step=1), xlabels)
    plt.xlabel(xlabel)
    #plt.yticks(np.arange(1, MAX_RANK+1, step=1))
    plt.ylabel("Rank")
    plt.grid(alpha=0.2)
    plt.gca().invert_yaxis()
    # Legend
    plt.legend(bbox_to_anchor=(1,1.02), loc="upper left")





#df.drop(columns=["passenger_loc_x"],inplace=True)
df = pd.read_csv("bump_simulation2.csv")
bumpchart(df, xlabels=["SFV",250,500,750,1000],xlabel="Simulations")
plt.show()
df = pd.read_csv("bump_all.csv")
bumpchart(df, xlabels=["SFV", 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],xlabel="Noise Strength")
plt.savefig('bump_features.pgf')
plt.show()
