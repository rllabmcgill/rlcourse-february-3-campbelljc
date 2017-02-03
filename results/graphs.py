import os, fileio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def graph_returns():
    plt.figure()
    plt.title('Avg. return over time')
    plt.ylim(0, 10)
    plt.xlim(0, 1000)
    plt.ylabel('Avg. returns')
    plt.xlabel('Number of state updates')
        
    for fname in os.listdir("."):
        if 'avg_returns' not in fname or '.txt' not in fname: continue
        
        y = fileio.read_line_list(fname)
        
        #window = np.ones(int(100))/float(100)
        #y_av = np.convolve(y, window, 'same')
        
        plt.plot([10*i for i in range(len(y))], y, label=fname.split("returns_")[1][:-4]) #_av[:-50])
        
    plt.legend()
    plt.savefig('avg_returns.png', bbox_inches='tight')

graph_returns()