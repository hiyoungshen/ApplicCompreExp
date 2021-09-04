# !
import numpy as np
import matplotlib.pyplot as plt

theta=np.pi/6
eps=0.000000001

t_interval = 0.0001
p=np.array([0, 0], dtype=np.float32)
v0=100
v=np.array([v0*np.cos(theta), v0*np.sin(theta)], dtype=np.float32)
m=1
k=0.3
f=-k*v**2
print(f)
g=np.array([0, -9.8], dtype=np.float32)

fig=plt.figure()
xs, ys=[], []

while p[1]>=-eps:
    # print(f, g)
    a=f+g
    v=v+a*t_interval
    p=p+v*t_interval
    xs.append(p[0])
    ys.append(p[1])
   
    print(p[0], p[1])
    f=-k*v**2
plt.plot(xs, ys)
plt.grid(axis="y")
plt.legend(['Parabola curve'])
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks(size = 9,rotation = 30)  # x轴标签旋转
plt.savefig('Parabola curve.png')
plt.show()

