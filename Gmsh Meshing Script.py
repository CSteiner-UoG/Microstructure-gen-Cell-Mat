# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 14:09:13 2021

@author: chase_vrj3scj
"""

# The Python API is entirely defined in the `gmsh.py' module (which contains the
# full documentation of all the functions in the API):
import gmsh
import sys
import numpy as np
import math

#%% Gmsh model creation and .step export
# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize()
#
# Next we add a new model named "t1" (if gmsh.model.add() is not called a new
# unnamed model will be created on the fly, if necessary):
gmsh.model.add("t1")
#import matrix of generated unit cell structure
unit_map = np.load('material_map.npy')
#create a 3x3 pattern of the unit cell
matrix_size = len(unit_map)

transform_matrix = np.zeros(((matrix_size*3), (matrix_size*3)), dtype = int)

for vc in range(0,3):
    for cy in range(0,matrix_size):
        for cx in range(0, matrix_size):
            transform_matrix[matrix_size * vc + cx, cy] = unit_map[cy,cx]
            transform_matrix[matrix_size * vc + cx, cy + matrix_size] = unit_map[cy,cx]
            transform_matrix[matrix_size * vc + cx, cy + (matrix_size*2)] = unit_map[cy,cx]


from matplotlib import pyplot as plt

plt.imshow(unit_map)
plt.set_cmap('gray_r')
plt.show()

plt.imshow(transform_matrix)
plt.show()

#find points where material exists
points = np.where(transform_matrix == 1)
points_list = list(zip(points[0] , points [1]))

#create openCASCADE geometry at the points where material exists
lc = 1e-2
lb = 1
i = 0
rect = []
for point in points_list:
    i += 1
    gmsh.model.occ.addRectangle(point[0], point[1], 0, lb, lb, tag=i)
    #gmsh.model.geo.addPoint(point[0], point[1], 0, lc)
    rect.append((0,i))

#clean up model and ensure valid geometry    
gmsh.model.occ.removeAllDuplicates()

gmsh.model.occ.fragment(objectDimTags = [], toolDimTags=[])
gmsh.model.occ.healShapes(dimTags=[], tolerance=1e-8, fixDegenerated=True, fixSmallEdges=True, fixSmallFaces=True, sewFaces=True, makeSolids=True)

#synchronize the model
gmsh.model.geo.synchronize()
gmsh.model.occ.synchronize()

#generate a 2D mesh
gmsh.model.mesh.generate(2)

#save to disk
gmsh.write("t1.msh")

#force gmsh to save all elements
gmsh.option.setNumber("Mesh.SaveAll", 1)

#output the model as a .step
gmsh.write('model.step')

#run gmsh GUI as long as popups are allowed
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

#close the Gmsh module 
gmsh.finalize()