import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

vecteur = np.array([[1],
                    [0]])
origine = np.array([[-2],
                    [4]])

droite = np.array([[1, 1],
                    [0, 2]])

def calculT(origineVecteur, vecteur, droite):
    droiteAS = np.array([[droite[0][0], origineVecteur[0][0]],
                   [droite[1][0], origineVecteur[1][0]]])
    vecteurAS = vDirecteur(droiteAS)
    normalAB = vNormal(vDirecteur(droite))
    return (-np.dot(vecteurAS, np.transpose(normalAB))[0][0]/(np.dot(np.transpose(normalAB), vecteur)[0][0]))

def toucheLaDroite(origineVecteur, vecteur, droite):
    t = calculT(origineVecteur, vecteur, droite)
    if t < 0:
        return None
    x = origineVecteur[0][0] + t * vecteur[0][0]
    y = origineVecteur[1][0] + t * vecteur[1][0]
    if (x <= max(droite[0, :][0], droite[0, :][1]) and x >= min(droite[0, :][0], droite[0, :][1])
    and y <= max(droite[1, :][0], droite[1, :][1]) and y >= min(droite[1, :][0], droite[1, :][1])):
        return np.array([[int(x)],
                         [int(y)]])
    else:
        return None

def vecteurEngendre(vecteur, vecteurNormal):
    if(np.dot(vecteur, np.transpose(vecteurNormal))[0][0] < 0):
        vecteurNormal = np.array([-vecteurNormal[0][0]],
                                 [-vecteurNormal[1][0]])

    print(np.dot(vecteur, np.transpose(vecteurNormal))[1][1])
    hypo = math.sqrt(np.dot(vecteur, np.transpose(vecteur))[0][0])
    adjacent = ( (abs(np.dot(vecteur, np.transpose(vecteurNormal))[1][1]))/math.sqrt(np.dot(vecteurNormal, np.transpose(vecteurNormal))[1][1]))
    angle = math.degrees(np.arccos(adjacent/hypo))
    print(angle)


def vNormal(vecteur):
    return np.array([[-vecteur[1, :][0]],
                     [vecteur[0, :][0]]])

def vDirecteur(droite):
    return np.array([[droite[:, 1][0]-droite[:, 0][0]],
                     [droite[:, 1][1]-droite[:, 0][1]]])

def construireVecteur(x, y):
    return np.array([[x],
                    [y]])

def construireDroite(x1, y1, x2, y2):
    return np.array([[x1, x2],
                       [y1, y2]])


print(toucheLaDroite(construireVecteur(0, 2), construireVecteur(1, -1), droite))
vecteurEngendre(construireVecteur(1, -1), vNormal(construireVecteur(1, 0)))

print(toucheLaDroite(construireVecteur(0, 1), construireVecteur(1, 0.5), construireDroite(2, 2, 2, 0)))


"""
x_data = []
y_data = []

fig, ax = plt.subplots()
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
line, = ax.plot(1, 5)

def animation_frame(i):
    angle = input()
    x_data.append(i*1)
    y_data.append(i*int(angle))

    line.set_xdata(x_data)
    line.set_ydata(y_data)
    return line,

animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 10, 0.1), interval=10)
plt.show()"""