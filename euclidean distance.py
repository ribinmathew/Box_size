import numpy as np

P1 = 400
Q1 = 300
P2 = 100
Q2 = 130

a = np.sqrt(np.sum((P1-P2)**2 + (Q1-Q2)**2))
print(a)