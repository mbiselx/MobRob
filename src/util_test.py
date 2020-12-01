import numpy as np
import matplotlib.pyplot as plt
import util

mu = np.array([[0], [0]])

sigma = np.array([[1, 0],
                  [0, 1]])

x = np.array([[-1,-1],[0,0],[1,1]]).T

print("mu", mu)
print("sigma", sigma)



print("\ntesting for single x")
print("x0", np.array([x[:,0]]).T)
z = util.mvnpdf2D(np.array([x[:,0]]).T, mu, sigma)
print("z:",z)


print("\ntesting for mulit x")
print("x:",x)
z = util.mvnpdf2D(x, mu, sigma)
print("z:",z)
