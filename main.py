import matplotlib.pyplot as plt
import numpy as np

vecteur = np.array([[1],
                    [0]])
origine = np.array([[-2],
                    [4]])

droite = np.array([[1, 5],
                   [1, 4]])

mur1 = np.array([[0, 0],
                 [0, 8]])
mur2 = np.array([[0, 10],
                 [0, 0]])
mur3 = np.array([[10, 10],
                 [2, 10]])
mur4 = np.array([[0, 10],
                 [10, 10]])
entree = np.array([[0, 0],
                 [8, 10]])
sortie = np.array([[10, 10],
                 [0, 2]])


"""
{ 00 0-10  10-0  10-10 }
 _ _ _ _ _ _ _ _ _ _
e                    |
e                    |
|                    |
|                    |
|                    |
|                    |
|                    |
|                    |
|                    s
| _ _ _ _ _ _ _ _ _ _s
"""

def calculT(origineVecteur, vecteur, droite):
    droiteAS = np.array([[droite[0][0], origineVecteur[0][0]],
                   [droite[1][0], origineVecteur[1][0]]])
    vecteurAS = vDirecteur(droiteAS)
    normalAB = vNormal(vDirecteur(droite))
    return (-np.dot(vecteurAS, np.transpose(normalAB))[0][0]/(np.dot(np.transpose(normalAB), vecteur)[0][0]))

def toucheLaDroite(origineVecteur, vecteur, droite):
    t = calculT(origineVecteur, vecteur, droite)
    if t < 0:
        return False
    x = origineVecteur[0][0] + t * vecteur[0][0]
    y = origineVecteur[1][0] + t * vecteur[1][0]
    return (x <= max(droite[0, :][0], droite[0, :][1]) and x >= min(droite[0, :][0], droite[0, :][1])
    and y <= max(droite[1, :][0], droite[1, :][1]) and y >= min(droite[1, :][0], droite[1, :][1]))

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


#print(toucheLaDroite(construireVecteur(0, 1), construireVecteur(1, 0.5), construireDroite(2, 2, 2, 0)))




k = 2*np.pi
w = 2*np.pi
dt = 0.01

x = np.linspace(0, 3, 151)

for i in range(500):
    t = i * dt
    y = np.cos(k*x - w*t)
    plt.plot(x, y)
    plt.pause(0.01) # pause avec duree en secondes

plt.show()