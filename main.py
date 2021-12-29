import math

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation

vecteur = np.array([[1],
                    [-0.5]])
origine = np.array([[-2],
                    [4]])

droite = np.array([[1, 1],
                    [0, 4]])

translation = None

# Trouve T
def calculT(origineVecteur, vecteur, droite):
    droiteAS = np.array([[droite[0][0], origineVecteur[0][0]],
                         [droite[1][0], origineVecteur[1][0]]])
    vecteurAS = vDirecteur(droiteAS)
    normalAB = vNormal(vDirecteur(droite))
    return ((-np.dot(np.transpose(normalAB), vecteurAS))[0][0] / (np.dot(np.transpose(normalAB), vecteur))[0][0])

# Retourne le point d'intersection
def toucheLaDroite(origineVecteur, vecteur, droite):
    t = calculT(origineVecteur, vecteur, droite)
    if t < 0:
        return None
    x = origineVecteur[0][0] + t * vecteur[0][0]
    y = origineVecteur[1][0] + t * vecteur[1][0]
    print(f"Point d'intersection: {x}, {y}")
    if (x <= max(droite[0, :][0], droite[0, :][1]) and x >= min(droite[0, :][0], droite[0, :][1])
    and y <= max(droite[1, :][0], droite[1, :][1]) and y >= min(droite[1, :][0], droite[1, :][1])):
        return np.array([[int(x)],
                         [int(y)]])
    else:
        return None


# Retourne l'angle du vecteur
def vecteurEngendre(vecteur, vecteurNormal):
    if(np.dot(np.transpose(vecteurNormal), vecteur ) < 0):
        vecteurNormal = np.array([[-vecteurNormal[0][0]],
                                 [-vecteurNormal[1][0]]])
    hypo = math.sqrt(np.dot(np.transpose(vecteur),vecteur))
    adjacent = ( (abs(np.dot( np.transpose(vecteurNormal),vecteur)))/math.sqrt(np.dot( np.transpose(vecteurNormal),vecteurNormal)))
    print(f"Angle: {math.degrees(np.arccos(adjacent / hypo))}")
    return math.degrees(np.arccos(adjacent/hypo))

# Retourne le vecteur après la rotation
def rotationVecteur(vect, impact, theta):
    rotation = theta
    x = vect[0][0]
    y = vect[1][0]

    if(x*y < 0 or (x == 0 and y > 0) or (y == 0 and x > 0) ):
        rotation = -theta

    boutDuVecteur = impact+vect

    # si pas besoin des coordonnées homogènes
    mat = np.array([[np.cos(rotation), -np.sin(rotation)],
                    [np.sin(rotation), np.cos(rotation)]])
    premierResultat = np.dot(mat,boutDuVecteur)
    if sontColinéaires(premierResultat, vect):
        mat2 = np.array([[np.cos(-rotation), -np.sin(-rotation)],
                        [np.sin(-rotation), np.cos(-rotation)]])
        return np.dot(mat2,boutDuVecteur)
    else:
        return premierResultat

def vNormal(vecteur):
    return np.array([[-vecteur[1, :][0]],
                     [vecteur[0, :][0]]])

def vDirecteur(droite):
    return np.array([[droite[:, 1][0] - droite[:, 0][0]],
                     [droite[:, 1][1] - droite[:, 0][1]]])

def normeVecteur(v1) :
    ca = v1[0]**2 + v1[1]**2
    return math.sqrt(ca)

def construireVecteur(x, y):
    return np.array([[x],
                     [y]])

def sontColinéaires(v1, v2):
    return v1[0][0]*v2[1][0] == v1[1][0]*v2[0][0]

def construireDroite(x1, y1, x2, y2):
    return np.array([[x1, x2],
                     [y1, y2]])


print(rotationVecteur(vecteur, toucheLaDroite(origine, vecteur, droite), vecteurEngendre(vecteur, vNormal(vecteur))))


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
plt.show()
droitetest = construireDroite(0, 2, 1, 1)
vec = vDirecteur(droitetest)
nv = normeVecteur(vec)
print(nv)

mur1 = construireDroite(0, 0, 0, 8)
mur2 = construireDroite(0, 0, 0, 8)
mur3 = construireDroite(0, 0, 0, 8)
mur4 = construireDroite(0, 0, 0, 8)
entree = construireDroite(0, 8, 0, 10)
sortie = construireDroite(10, 0, 10, 2)


def visu_point(matPoint, style):
    # matPoint contient les coordonnées des points
    x = matPoint[0, :]
    y = matPoint[1, :]
    plt.plot(x, y, style)


def visu_segment(P1, P2, style):
    # attention P1 et P2 sont des tableaux (2,1)
    matP = np.concatenate((P1, P2), 1)
    visu_point(matP, style)




def getPointFromMatrice(numPoint, matricePts):
    pt = np.array([[matricePts[0][0], matricePts[1][0]],
                   [matricePts[0][1], matricePts[1][1]]])
    return pt[numPoint-1]



x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
z = 2*x+4
plt.plot(z)
plt.show()

"""

