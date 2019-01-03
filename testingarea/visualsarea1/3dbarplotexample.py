import matplotlib.pyplot as pt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

fig = pt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection = '3d')

for c,z in zip(['r','g','b','y'],[30,20,10,0]):
    xs = np.arange(20)
    ys = np.random.rand(20)

cs = [c]*len(xs)
cs[0] = 'c'

ax.bar(xs, ys, zs = z, zdir = 'y', color = cs, alpha = 0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

pt.show()
#technically doesn't completely work