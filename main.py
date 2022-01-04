import math

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation

vecteur = np.array([[1],[-0.5]])
origine = np.array([[0],[9]])
droite = np.array([[-2,10 ],[-2, 10]])

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

    if (x <= max(droite[0, :][0], droite[0, :][1]) and x >= min(droite[0, :][0], droite[0, :][1])
    and y <= max(droite[1, :][0], droite[1, :][1]) and y >= min(droite[1, :][0], droite[1, :][1])):
        return np.array([[(x)],
                         [(y)]])
    else:
        return None


# Retourne l'angle du vecteur
def vecteurEngendre(vecteur, vecteurNormal):
    if(np.dot(np.transpose(vecteurNormal), vecteur ) > 0):
        vecteurNormal = np.array([[-vecteurNormal[0][0]],
                                 [-vecteurNormal[1][0]]])
    hypo = math.sqrt(np.dot(np.transpose(vecteur),vecteur))
    adjacent = ((abs(np.dot( np.transpose(vecteurNormal),vecteur)))/math.sqrt(np.dot( np.transpose(vecteurNormal),vecteurNormal)))

    return math.degrees(np.arccos(adjacent/hypo))

# Retourne le vecteur apres la rotation
def rotationVecteur(vect, impact, theta):
    if impact is None :
        return None
    rotation = theta
    x = vect[0][0]
    y = vect[1][0]

    if(x*y < 0 or (x == 0 and y > 0) or (y == 0 and x > 0) ):
        rotation = -theta

    boutDuVecteur = impact+vect

    #matrice rotation
    mat = np.array([[np.cos(rotation), -np.sin(rotation)],
                    [np.sin(rotation), np.cos(rotation)]])
    premierResultat = np.dot(mat,boutDuVecteur)
    if sontColineaires(premierResultat, vect):
        mat2 = np.array([[np.cos(-rotation), -np.sin(-rotation)],
                        [np.sin(-rotation), np.cos(-rotation)]])
        return np.dot(mat2,boutDuVecteur),impact
    else:
        return premierResultat,impact

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

def sontColineaires(v1, v2):
    return v1[0][0]*v2[1][0] == v1[1][0]*v2[0][0]


def construireDroite(x1, y1, x2, y2):
    return np.array([[x1, x2],
                     [y1, y2]])







mur1 = construireDroite(0, 0, 0, 8)
mur2 = construireDroite(0, 10, 10, 10)
mur3 = construireDroite(10, 10, 10, 2)
mur4 = construireDroite(10, 0, 0, 0)
entree = construireDroite(0, 8, 0, 10)
sortie = construireDroite(10, 0, 10, 2)

listemur = [mur1,mur2,mur3, mur4]
endCond = [entree,sortie]
segx1 = construireDroite(4, 9.5, 4, 2)
segx2 = construireDroite(2.2, 7.27, 3.56, 4.31)
segx3 = construireDroite(6.96, 5.43, 7.96, 3.21)
segx4 = construireDroite(3.42, 2.19, 4.94, 0.55)

map = [segx1,segx2,segx3,segx4]

vdir = vDirecteur(construireDroite(0,9,1,8.5))
# rotationVecteur(vect, impact, theta):

print(rotationVecteur(vNormal(vdir), toucheLaDroite(origine, vdir, segx1), vecteurEngendre(vdir, vNormal(vDirecteur(segx1)))))

def construreSegments (liste) :
    for mur in liste:
        x = np.linspace(mur[0, :][0], mur[0, :][1], 2)
        y = np.linspace(mur[1, :][0], mur[1, :][1], 2)
        plt.scatter(x, y)
        plt.plot(x, y)

construreSegments(listemur)
construreSegments(map)


def construireDroite(x1, y1, x2, y2):
    return np.array([[x1, x2],
                     [y1, y2]])

plt.quiver(0, 9, 1, -0.5, angles = 'xy', scale_units = 'xy', scale = 1)


plt.show()




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



def visu_point(matPoint, style):
    # matPoint contient les coordonnees des points
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
