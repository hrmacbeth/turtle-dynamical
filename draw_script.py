# This file contains a bunch of lightly-disguised matplotlib functions, for use in teaching a math
# course in which the precise details of the plotting mechanism are not important.
#
# In particular, the first half of the file gives a minimalist turtle graphics implementation.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def init():
    global p
    global q
    p = [[0.]]
    q = [[0.]]

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

def plot_vector_field(xfun, yfun, color='lightsteelblue'):
    """Plot a vector field, specified by functions for the x- and y- components of the vector field.
    Wrapper for matplotlib.pyplot.quiver.
    """
    x,y = np.meshgrid(np.linspace(-50,50,20),np.linspace(-50,50,20))
    plt.quiver(x, y, list(map(xfun, x, y)), list(map(yfun, x, y)), color=color)

def plot_points(ax, axis_scale, X_solid, y_solid, X_semitransparent=[], y_semitransparent=[]):
    """Plot two collections of points (for example, training data and test data) in the plane.  
    Each point must come with a value, to be used (with the red-blue colourmap) to determine its colour.
    The first collection is plotted in solid colours, the second semi-transparently.
    Wrapper for matplotlib.pyplot.scatter.
    """
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # Plot the training points
    ax.scatter(X_solid[:, 0], X_solid[:, 1], c=y_solid, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    if X_semitransparent != []:
        ax.scatter(X_semitransparent[:, 0], X_semitransparent[:, 1], c=y_semitransparent, 
                   cmap=cm_bright, alpha=0.6, edgecolors='k')
    ax.set_xlim(-axis_scale, axis_scale)
    ax.set_ylim(-axis_scale, axis_scale)
    ax.set_xticks(())
    ax.set_yticks(())

def plot_3d(X, Y, Z):
    """Plot a surface, specified by a collection of (x, y, z)-co-ordinates.
    Wrapper for matplotlib.pyplot.scatter.
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
