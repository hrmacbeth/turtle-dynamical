import matplotlib.pyplot as plt
import numpy as np

def init():
    global p
    global q
    p = [[0]]
    q = [[0]]

def move(dx, dy):
    global p
    global q
    i = len(p) - 1
    p[i].append(p[i][len(p[i]) - 1] +  dx)
    q[i].append(q[i][len(q[i]) - 1] +  dy)

def jump_to(x, y):
    global p
    global q
    p.append([x])
    q.append([y])

def x():
    global p
    l = len(p) - 1
    m = len(p[l]) - 1
    return p[l][m]

def y():
    global q
    l = len(q) - 1
    m = len(q[l]) - 1
    return q[l][m]

def draw(**kwargs):
    global p
    global q
    plt.rcParams['figure.figsize'] = [6, 6]
    l = len(p)
    for i in range(l):
        plt.plot(p[i], q[i], **kwargs)
    plt.xlim(-50,50)
    plt.ylim(-50,50)
    plt.show()

def draw_with_vector_background(xfun, yfun, **kwargs):
    x,y = np.meshgrid(np.linspace(-50,50,20),np.linspace(-50,50,20))
    plt.quiver(x, y, list(map(xfun, x, y)), list(map(yfun, x, y)), color='lightsteelblue')
    draw(**kwargs)
