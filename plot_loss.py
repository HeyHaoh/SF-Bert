import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a = pd.read_csv("loss.csv", delimiter=",", usecols=[0])
b = pd.read_csv("loss.csv", delimiter=",", usecols=[1])

plt.figure(figsize=(40,5))
plt.plot(a,b)
# plt.show()
plt.savefig('loss_mine.png')