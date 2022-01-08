import math

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation

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
    if (origineVecteur[0][0] <= max(droite[0, :][0], droite[0, :][1]) and origineVecteur[0][0]  >= min(droite[0, :][0], droite[0, :][1])
            and origineVecteur[1][0]  <= max(droite[1, :][0], droite[1, :][1]) and origineVecteur[1][0]  >= min(droite[1, :][0], droite[1, :][1])):
        return None
    t = calculT(origineVecteur, vecteur, droite)
    if t < 0:
        return None
    x = origineVecteur[0][0] + t * vecteur[0][0]
    y = origineVecteur[1][0] + t * vecteur[1][0]
    if(x < 0.0001):
        x = 0
    if(y < 0.0001):
        y = 0
    if (x <= max(droite[0, :][0], droite[0, :][1]) and x >= min(droite[0, :][0], droite[0, :][1])
    and y <= max(droite[1, :][0], droite[1, :][1]) and y >= min(droite[1, :][0], droite[1, :][1])):
        return np.array([[(x)],
                         [(y)]])
    else:
        return None


"""def impactFinal(impact, entree, sortie):
    if((impact[0][0] == sortie[0][0] and impact[1][0] <= 2 ) or
    (impact[0][0] == entree[0][0] and impact[1][0] >= 8 ))
        return True
    else:   3
        return False"""



        
"""# Retourne l'angle du vecteur
def vecteurEngendre(vecteur, vecteurNormal):
    if(np.dot(np.transpose(vecteurNormal), vecteur ) > 0):
        vecteurNormal = np.array([[-vecteurNormal[0][0]],
                                 [-vecteurNormal[1][0]]])
    hypo = math.sqrt(np.dot(np.transpose(vecteur),vecteur))
    adjacent = ((abs(np.dot( np.transpose(vecteurNormal),vecteur)))/math.sqrt(np.dot( np.transpose(vecteurNormal),vecteurNormal)))

    return math.degrees(np.arccos(adjacent/hypo))
"""
# Retourne le vecteur apres la rotation
def rotationVecteur(vect, impact, vecteurNormal):
    if impact is None :
        return None
    if(np.dot(np.transpose(vecteurNormal), vecteur ) > 0):
        vecteurNormal = np.array([[-vecteurNormal[0][0]],
                                 [-vecteurNormal[1][0]]])
    x = 2 * (np.dot(np.transpose(vect), vecteurNormal)/(normeVecteur(vecteurNormal)**2))
    x = x[0][0]
    return np.array([[vect[0][0]-(x*vecteurNormal[0][0])],
                     [vect[1][0]-(x*vecteurNormal[1][0])]])

def vNormal(vecteur):
    n = np.array([[-vecteur[1, :][0]],
                     [vecteur[0, :][0]]])
    return np.array([[-vecteur[1, :][0]],
                     [vecteur[0, :][0]]])



def vDirecteur(droite):
    return np.array([[droite[:, 1][0] - droite[:, 0][0]],
                     [droite[:, 1][1] - droite[:, 0][1]]])

def normeVecteur(v1) :
    ca = np.dot(np.transpose(v1), v1)
    return math.sqrt(ca)

def construireVecteur(x, y):
    return np.array([[x],
                     [y]])

def sontColineaires(v1, v2):
    return v1[0][0]*v2[1][0] == v1[1][0]*v2[0][0]


def construireDroite(x1, y1, x2, y2):
    return np.array([[x1, x2],
                     [y1, y2]])

def get_key(val, liste):
    for key, value in liste.items():
         if val == value:
             return key

def trouvePlusProche(origine, liste):
    clésDistance = []
    valeursDistances = []
    i = 0
    for point in liste:
        if point is not None :
                valeursDistances.append((math.sqrt(((origine[0][0] - point[0][0])**2) + ((origine[1][0] - point[1][0])**2))))
                clésDistance.append(i)
        i += 1

    for v in valeursDistances:
        if v <= 0.01:
            valeursDistances.remove(v)

    plusPetit = valeursDistances[0]

    for d in valeursDistances:
        if d <= plusPetit:
            plusPetit = d

    return liste[clésDistance[valeursDistances.index(plusPetit)]], clésDistance[valeursDistances.index(plusPetit)]
    



mur1 = construireDroite(0, 0, 0, 8)
mur2 = construireDroite(0, 10, 10, 10)
mur3 = construireDroite(10, 10, 10, 2)
mur4 = construireDroite(10, 0, 0, 0)
entree = construireDroite(0, 8, 0, 10)
sortie = construireDroite(10, 0, 10, 2)

listemur = [mur1,mur2,mur3, mur4]
endCond = [entree,sortie]
segx1 = construireDroite(5, 10, 5, 7)
segx2 = construireDroite(2, 5, 0, 5)
segx3 = construireDroite(7, 5, 7, 1)
segx4 = construireDroite(8, 6, 10, 6)

map = [segx1,segx2,segx3,segx4]

tousLesMurs = map+listemur

vdir = vDirecteur(construireDroite(0,9,1,9))
# rotationVecteur(vect, impact, theta):

vecteur = np.array([[1],[2]])
origine = np.array([[0],[9]])
impact = np.array([[-1], [-99999]]) 

for i in range(10):
    listeImpactes = []

    plt.quiver(origine[0][0], origine[1][0], vecteur[0][0], vecteur[1][0], angles = 'xy', scale_units = 'xy', scale = 1)

    for segment in tousLesMurs:
        listeImpactes.append(toucheLaDroite(origine, vecteur, segment))

    impact, indiceMur = trouvePlusProche(origine, listeImpactes)

    vecteur = rotationVecteur(vecteur, impact, vNormal(vDirecteur(tousLesMurs[indiceMur])))

    origine = impact



def construireSegments (liste) :
    for mur in liste:
        x = np.linspace(mur[0, :][0], mur[0, :][1], 2)
        y = np.linspace(mur[1, :][0], mur[1, :][1], 2)
        plt.scatter(x, y)
        plt.plot(x, y)

construireSegments(listemur)
construireSegments(map)


def construireDroite(x1, y1, x2, y2):
    return np.array([[x1, x2],
                     [y1, y2]])



plt.show()



