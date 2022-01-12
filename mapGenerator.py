import math
import time

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation



def construireDroite(x1, y1, x2, y2):
    return np.array([[x1, x2],
                     [y1, y2]])
#MAP NUMERO 1
segx1_MAP1 = construireDroite(1, 3, 4, 1)
segx2_MAP1 = construireDroite(7, 8, 5, 3)
segx3_MAP1 = construireDroite(7, 1, 7, 3)
segx4_MAP1 = construireDroite(6, 8, 3, 9)
map1 = [segx1_MAP1, segx2_MAP1, segx3_MAP1, segx4_MAP1]

#MAP NUMERO 2
segx1_MAP2 = construireDroite(2, 10, 2, 7)
segx2_MAP2 = construireDroite(0, 5, 3, 5)
segx3_MAP2 = construireDroite(3.86, 7.73, 5, 5.59)
segx4_MAP2 = construireDroite(3.86, 7.73, 6, 9)
segx5_MAP2 = construireDroite(5,5.59,6.98,5.69)
segx6_MAP2 = construireDroite(4,4,6,2)
segx7_MAP2 = construireDroite(2,3,2,2)
segx8_MAP2 = construireDroite(2,2,3,2)
segx9_MAP2 = construireDroite(6,2,8,4)
segx10_MAP2 = construireDroite(8,10,10,8)
map2 = [segx1_MAP2, segx2_MAP2, segx3_MAP2, segx4_MAP2, segx5_MAP2,
        segx6_MAP2, segx7_MAP2, segx8_MAP2, segx9_MAP2, segx10_MAP2]

#MAP NUMERO 3
segx1_MAP3 = construireDroite(8, 9.5, 9, 9.5)
segx2_MAP3 = construireDroite(8, 9, 9, 9)
segx3_MAP3 = construireDroite(9, 9.5, 9.8, 8.7)
segx4_MAP3 = construireDroite(9, 9, 9.3, 8.7)
segx5_MAP3 = construireDroite(9.3, 8.7, 9.3, 1.7)
segx6_MAP3 = construireDroite(9.8, 8.7, 9.8, 2)
segx7_MAP3 = construireDroite(1, 2, 3, 2)
#segx8_MAP3 = construireDroite(9.7, 1.5, 9.7, 1.7)
segx9_MAP3 = construireDroite(9.3,1.7,9.9,1)
segx10_map3= construireDroite(4.5,9,4.5,9.7)
segx11_map3 = construireDroite(2,6,5,6)
segx12_map3 = construireDroite(4,4,8,4)
segx13_map3 = construireDroite(8,2,8,0.3)

map3 = [segx1_MAP3, segx2_MAP3, segx3_MAP3, segx4_MAP3,segx5_MAP3,
        segx6_MAP3,segx7_MAP3,segx9_MAP3,segx10_map3,
        segx11_map3, segx12_map3, segx13_map3]

def getmap(value):
    if value == 1:
        return map1,15
    if value == 2:
        return map2,40
    if value == 3:
        return map3,30
