import os
import numpy as np

prices = []
with open(os.path.join(os.path.dirname(__file__), 'input.txt')) as f:
    for line in f:
        prices.append(float(line.strip()))

print(" mean = ",float(sum(prices)) / len(prices))
print(" np mean = ",np.mean(np.array(prices)))
print(" np std = ",np.std(np.array(prices)))

