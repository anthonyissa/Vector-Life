import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

vecteur = np.array([[1],
                    [0]])
origine = np.array([[-2],
                    [4]])

droite = np.array([[1, 5],
                   [1, 4]])


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
    return (-np.dot(vecteurAS, np.transpose(normalAB))[0][0] / (np.dot(np.transpose(normalAB), vecteur)[0][0]))


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
    return np.array([[droite[:, 1][0] - droite[:, 0][0]],
                     [droite[:, 1][1] - droite[:, 0][1]]])


def construireVecteur(x, y):
    return np.array([[x],
                     [y]])


def construireDroite(x1, y1, x2, y2):
    return np.array([[x1, x2],
                     [y1, y2]])


mur1 = construireDroite(0, 0, 0, 8)
mur2 = construireDroite(0, 0, 0, 8)
mur3 = construireDroite(0, 0, 0, 8)
mur4 = construireDroite(0, 0, 0, 8)


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


def visu_point(matPoint, style):
    # matPoint contient les coordonnées des points
    x = matPoint[0, :]
    y = matPoint[1, :]
    plt.plot(x, y, style)


def visu_segment(P1, P2, style):
    # attention P1 et P2 sont des tableaux (2,1)
    matP = np.concatenate((P1, P2), 1)
    visu_point(matP, style)


def mat_rotation(theta):
    # si pas besoin des coordonnées homogènes
    mat = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    return mat


def getPointFromMatrice(numPoint, matricePts):
    pt = np.array([[matricePts[0][0], matricePts[1][0]],
                   [matricePts[0][1], matricePts[1][1]]])
    if (numPoint == 1):
        return pt[0]
    if (numPoint == 2):
        return pt[1]


x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
z = 2*x+4
plt.plot(z)
plt.show()


