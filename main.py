import math
import time
import mapGenerator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation

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
    if(x < 0.01):
        x = 0
    if(y < 0.01):
        y = 0
    x = round(x, 2)
    y = round(y, 2)
    if (x <= max(droite[0, :][0], droite[0, :][1]) and x >= min(droite[0, :][0], droite[0, :][1])
    and y <= max(droite[1, :][0], droite[1, :][1]) and y >= min(droite[1, :][0], droite[1, :][1])):
        return np.array([[(x)],
                         [(y)]])
    else:
        return None

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

def distanceDeuxPoints(x1, y1, x2, y2):
    return math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

def trouvePlusProche(origine, liste):
    clésDistance = []
    valeursDistances = []
    i = 0
    for point in liste:
        if point is not None :
                valeursDistances.append((math.sqrt(((origine[0][0] - point[0][0])**2) + ((origine[1][0] - point[1][0])**2))))
                clésDistance.append(i)
        i += 1

    plusPetit = valeursDistances[0]

    for d in valeursDistances:
        if d <= plusPetit:
            plusPetit = d

    return liste[clésDistance[valeursDistances.index(plusPetit)]], clésDistance[valeursDistances.index(plusPetit)]

def construireSegments (liste) :
    for mur in liste:
        x = np.linspace(mur[0, :][0], mur[0, :][1], 2)
        y = np.linspace(mur[1, :][0], mur[1, :][1], 2)
        plt.scatter(x, y)
        plt.plot(x, y,color ='black')
#mapping
mur1 = construireDroite(0, 0, 0, 8)
mur2 = construireDroite(0, 10, 10, 10)
mur3 = construireDroite(10, 10, 10, 2)
mur4 = construireDroite(10, 0, 0, 0)
#premier affichage des murs
listeDesMur = [mur1, mur2, mur3, mur4]
construireSegments(listeDesMur)
plt.pause(0.1)

entree = construireDroite(0, 8, 0, 10)
sortie = construireDroite(10, 0, 10, 2)
endCond = [entree, sortie]



print("SELECTIONNEZ UNE CARTE : 1 2 OU 3 ")
map,difficulté = mapGenerator.getmap(int(input()))
tousLesMurs = map + listeDesMur + endCond
mursEtCheminAffichés = map + listeDesMur
vdir = vDirecteur(construireDroite(0,9,1,9))
# rotationVecteur(vect, impact, theta):


construireSegments(mursEtCheminAffichés)
plt.pause(0.1)
print("ENTREZ LA PENTE DU VECTEUR")
PENTE = input()

vecteur = np.array([[1],[float(PENTE)]])
origine = np.array([[0.2],[9]])
impact = np.array([[0.2], [9]])
ancienOrgine = origine

nombreImapcts = 0


def environEgal(val1,val2):
    #permet de d'arrondir les valeurs obenue qui sont parfois trop imprécises.
    return (math.sqrt((val1-val2)**2) < 0.1)


def verficarteurDeFin(impact):
    if (environEgal(0,impact[0]) and impact[1]>8 and impact[1]<10) :
        construireSegments(mursEtCheminAffichés)
        plt.title("YOU LOST IN " + str(nombreImapcts-1)+" IMPACTS")
    if (nombreImapcts>=difficulté):
        construireSegments(mursEtCheminAffichés)
        plt.title("YOU LOST IN" + str(nombreImapcts-1)+" IMPACTS")
    if ((environEgal(10,impact[0]) and impact[1]>0 and impact[1]<2)) :
        construireSegments(mursEtCheminAffichés)
        plt.title("YOU WIN IN " + str(nombreImapcts)+" IMPACTS")
    return not ((environEgal(0,impact[0]) and impact[1]>8 and impact[1]<10) or ((environEgal(10,impact[0]) and impact[1]>0 and impact[1]<2)) or nombreImapcts>difficulté)



while True:

    listeImpactes = []

    for segment in tousLesMurs:
        listeImpactes.append(toucheLaDroite(origine, vecteur, segment))
    impact, indiceMur = trouvePlusProche(origine, listeImpactes)

    plt.quiver(origine[0][0], origine[1][0], vecteur[0][0], vecteur[1][0], angles='xy', scale_units='xy',color="blue",
               scale=(2))

    plt.plot([origine[0][0],impact[0][0]],[origine[1][0],impact[1][0]])

    x = construireDroite(origine[0][0],impact[0][0],origine[1][0],impact[1][0])
    #mursEtCheminAffichés.append(x)

    construireSegments(mursEtCheminAffichés)
    plt.pause(0.25)
    origine = impact
    if (verficarteurDeFin(impact)==False):
        break
    nombreImapcts += 1
    plt.suptitle(str(nombreImapcts) + " IMPACTS!")
    vecteur = rotationVecteur(vecteur, impact, vNormal(vDirecteur(tousLesMurs[indiceMur])))


#affichage

def construireDroite(x1, y1, x2, y2):
    return np.array([[x1, x2],
                     [y1, y2]])


plt.show()

