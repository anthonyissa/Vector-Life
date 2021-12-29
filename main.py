import math

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation

translationDuMoment = np.array([[0],
                            [0]])
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
    return (-np.dot(vecteurAS, np.transpose(normalAB))[0][0] / (np.dot(np.transpose(normalAB), vecteur)[0][0]))


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
    #print(vecteurNormal)
    if(np.dot(np.transpose(vecteurNormal), vecteur ) < 0):
        vecteurNormal = np.array([[-vecteurNormal[0][0]],
                                 [-vecteurNormal[1][0]]])
    #print(vecteurNormal)
    #print(np.dot(vecteur, np.transpose(vecteur))[1][1])
    hypo = math.sqrt(np.dot(np.transpose(vecteur),vecteur))




    adjacent = ( (abs(np.dot( np.transpose(vecteurNormal),vecteur)))/math.sqrt(np.dot( np.transpose(vecteurNormal),vecteurNormal)))
    print(np.arccos(adjacent / hypo))
    print('prouitcul')
    angle = math.degrees(np.arccos(adjacent/hypo))
    print(angle)



def vNormal(vecteur):
    return np.array([[-vecteur[1, :][0]],
                     [vecteur[0, :][0]]])

def vDirecteur(droite):
    return np.array([[droite[:, 1][0] - droite[:, 0][0]],
                     [droite[:, 1][1] - droite[:, 0][1]]])

def normeVecteur(v1) :
    print(v1)
    ca = v1[0]**2 + v1[1]**2
    return math.sqrt(ca)

def construireVecteur(x, y):
    return np.array([[x],
                     [y]])

def construireDroite(x1, y1, x2, y2):
    return np.array([[x1, x2],
                     [y1, y2]])


#print(toucheLaDroite(construireVecteur(0, 2), construireVecteur(1, -1), droite))
vecteurEngendre(construireVecteur(1, -1), vNormal(construireVecteur(1, 0)))

#print(toucheLaDroite(construireVecteur(0, 1), construireVecteur(1, 0.5), construireDroite(2, 2, 2, 0)))

def mat_rotation(theta):
    # si pas besoin des coordonnées homogènes
    mat = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    return mat


def mat_translationOrigine(pointimpact,vect) :
    translationDuMoment = -v1
    print(translationDuMoment)


mat_translationOrigine(construireVecteur(2,2))
def rotationApresImpact(v1):


    vectR = mat_rotation()

    return vectR
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
"""
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


