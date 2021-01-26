# -*- coding: utf-8 -*-

"""
Created on Thu Jan  7 14:09:13 2021

@author: chase_vrj3scj
"""

#%% import libraries
import sys
import numpy as np
import random
import matplotlib as mpl
import gmsh

#%% material generation
random.seed(a = None)

#specify dimension of matrix for bitmap, matrix shape is square so only one
#edge length needs specified
matrix_size = 10

#make matrix of zeros to stand as the bitmap
bitmap_initial = np.zeros((matrix_size,matrix_size), dtype=int)
#print(bitmap[40,20])

#since the bitmap matrix is only populated by 1's and 0's at this point,
#porosity = the arithmetic mean of the flattened matrix
bitmap_avg = porosity = np.mean(bitmap_initial)

print(porosity)


#seed the bitmap matrix with edge material seeds
for i in range(5):
    x = random.randint(0,matrix_size - 1)    
    y = random.randint(0,matrix_size - 1)
    
    selector = random.randint(1,5)
    
    if selector == 1:
        bitmap_initial[x,0] = 1
    elif selector == 2:    
        bitmap_initial[0,y] = 1
    elif selector == 3:    
        bitmap_initial[x,matrix_size - 1] = 1
    elif selector == 4:
        bitmap_initial[matrix_size - 1, y] = 1
    elif selector >= 5:
        bitmap_initial[x,y] = 1
    

#specify desired porosity of structure
porosity_desired = .25

#with matrix of 1's, inverse of desired porosity will give the target mean of the
#matrix
pd_inverse = round(1 - porosity_desired , 3)

print(pd_inverse)
print(np.mean(bitmap_initial))

#show seeded bitmap
from matplotlib import pyplot as plt
plt.imshow(bitmap_initial)
plt.set_cmap('gray_r')
plt.show()


#show the locations of the seeded material
mat_result = np.where(bitmap_initial == 1)
mat_index = list(zip(mat_result[0], mat_result[1]))

for coord in mat_index:
    print(coord)

bitmap_fill = bitmap_initial

#grow/fill voids until desired porosity is met
while porosity != pd_inverse:
    #to propagate material, need index of where they are so that the adjacent 
    #cells can be selected randomly to become material
    material = np.where(bitmap_fill == 1)
    index = list(zip(material[0], material[1]))
    #grow material if current porosity is less than desired
    if porosity < pd_inverse:
        #for each located void, randomly select an adjacent cell using j for
        #horizontal location and k for vertical
        for location in index:
            j = random.randint(-1,1)
            k = random.randint(-1,1)
            
            #check that a diagonal hasn't been specified, regenerate until 
            #adjacent
            while (j == k and (j == 1 or j == -1)):
                j = random.randint(-1,1)
                k = random.randint(-1,1)
                    
            #make sure new voids are within bounds of bitmap matrix
            if j == 1 and location[0] == matrix_size - 1:
                    j = random.randint(-1,0)
            if j == -1 and location[0] == 0:
                    j = random.randint(0,1)
            if k == 1 and location[1] == matrix_size - 1:
                    k = random.randint(-1,0)
            if k == -1 and location[1] == 0:
                    k = random.randint(0,1)
            
            #create material in specified cell        
            bitmap_fill[location[0] + j,location[1] + k] = 1
            
            #check porosity again for the 'for' loop and exit if reached
            porosity = np.mean(bitmap_fill)
            if porosity == pd_inverse:
                break       
       
        #clear material if current porosity is more than desired
    elif porosity > pd_inverse:
            
            #for each located void, randomly select an adjacent cell using j 
            #for horizontal location and k for vertical
            for location in index:
                j = random.randint(-1,1)
                k = random.randint(-1,1)
                
                #check that a diagonal hasn't been specified, regenerate until
                #adjacent
                while (j == k or j == -k):
                    j = random.randint(-1,1)
                    k = random.randint(-1,1)
                        
                #make sure new voids are within bounds of bitmap matrix
                if j == 1 and location[0] == matrix_size - 1:
                        j = random.randint(-1,0)
                if j == -1 and location[0] == 0:
                        j = random.randint(0,1)
                if k == 1 and location[1] == matrix_size - 1:
                        k = random.randint(-1,0)
                if k == -1 and location[1] == 0:
                        k = random.randint(0,1)
                
                #fill void in specified cell        
                bitmap_fill[location[0] + j,location[1] + k] = 0
                
                #check porosity again for the 'for' loop and exit if reached
                porosity = np.mean(bitmap_fill)
                if porosity == pd_inverse:
                    break            
    #check porosity for while loop
    porosity = np.mean(bitmap_fill)

#output the final porosity of the bitmap and show the bitmap 
print(porosity)
plt.imshow(bitmap_fill)
plt.show()

#%% create global body map
def global_assembly(bitmap):
    
    #reflect matrix over x axis
    bitmap_xreflect = np.zeros((matrix_size,matrix_size), dtype=int)
    for cx in range(0,matrix_size):
        for cy in range(0,matrix_size):
            bitmap_xreflect[cy ,cx  ] = bitmap[matrix_size - 1 - cy , cx ]
    
    
    plt.imshow(bitmap_xreflect)
    plt.show()
    
    #reflect matrix over y axis
    bitmap_yreflect = np.zeros((matrix_size,matrix_size), dtype=int)
    for cy in range(0,matrix_size):
        for cx in range(0,matrix_size):
            bitmap_yreflect[cy ,cx  ] = bitmap[ cy , matrix_size - 1 - cx ]
         
    plt.imshow(bitmap_yreflect)
    plt.show()
    
    #reflect the yreflect over x to get final quadrant
    bitmap_yxreflect = np.zeros((matrix_size,matrix_size), dtype=int)
    for cx in range(0,matrix_size):
        for cy in range(0,matrix_size):
            bitmap_yxreflect[cy ,cx  ] = bitmap_yreflect[matrix_size - 1 - cy , cx ]
    
    plt.imshow(bitmap_yxreflect)
    plt.show()
    
    #create global body matrix
    globalmap = np.zeros((matrix_size *2, matrix_size *2), dtype=int)
    
    
    
    for cx in range(0,matrix_size):
        for cy in range(0,matrix_size):
            globalmap[cy ,cx  ] = bitmap[cy , cx ]
                    
    for cx in range(0,matrix_size):
        for cy in range(0,matrix_size):
            globalmap[cy + matrix_size ,cx  ] = bitmap_xreflect[cy , cx ]
     
    for cx in range(0,matrix_size):
        for cy in range(0,matrix_size):
            globalmap[cy ,cx + matrix_size ] = bitmap_yreflect[cy , cx ]
        
    for cx in range(0,matrix_size):
        for cy in range(0,matrix_size):
            globalmap[cy + matrix_size,cx + matrix_size] = bitmap_yxreflect[cy , cx ]
        
    return globalmap

unit_map = global_assembly(bitmap_fill)
    
print(np.mean(unit_map))
plt.imshow(unit_map)
plt.show()

porosity_final = 1 - porosity

print(porosity_final)

#%% find floaters and reposition
#as this will be called upon multiple times potentially, will be made into func
def position_weighting(structuremap):
    position_matrix = np.zeros((len(structuremap), len(structuremap)), dtype=int)

    ones_body = np.where(structuremap == 1)
    index_body = list(zip(ones_body[0], ones_body[1]))

    #currently the for loops re-add the value inside the position matrix cell twice at least
    #(fixed)
    
    #skip variable initialized so the if conditional is populated
    skip = 0
    
    for solids in index_body:
        #use if conditionals to define upper and lower bounds of range values
        #to remain in bounds of matrix; lb_v/h is inclusive -- ub_v/h is exlcusive
        #top edge boundary 
        if solids[1] == 0:
            lb_v = 0
            ub_v = 2
        #bottom edge boundary
        elif solids[1] == (matrix_size*2) - 1:
            lb_v = -1
            ub_v = 1
        elif solids[1] != 0 or (matrix_size*2) - 1:
            lb_v = -1
            ub_v = 2
        for v in range(lb_v, ub_v):
        #add vertical adjacent values, use if to check within bounds of matrix
            position_matrix[solids[0], solids[1]] = position_matrix[solids[0], solids[1]] + structuremap[solids[0], solids[1] + v]
        #add horizontal adjacent values, use if to check within bounds of matrix
        #left edge boundary
        if solids[0] == 0:
            lb_h = 0
            ub_h = 2 
        elif solids[0] == (matrix_size*2) - 1:
            lb_h = -1
            ub_h = 1
        elif solids[0] != 0 or (matrix_size*2) - 1:
            lb_h = -1
            ub_h = 2
        for h in range(lb_h, ub_h):
            if h == 0:
                skip += 1
            else:   
                position_matrix[solids[0] , solids[1]] = position_matrix[solids[0] , solids[1]] + structuremap[solids[0] + h , solids[1]]
    
    return position_matrix


weighted_matrix = position_weighting(unit_map)
plt.imshow(weighted_matrix)
plt.show()

position_weighted_initial = weighted_matrix

ones_floaters = np.where(weighted_matrix == 1)
ones_index = list(zip(ones_floaters[0], ones_floaters[1]))

connected_points = np.where(weighted_matrix >= 2)
connected_index = list(zip(connected_points[0], connected_points[1]))
count = 0
while len(ones_index) > 0:
    for floaters in ones_index:
        unit_map[floaters[0] , floaters[1]] = 0
        
        connections = connected_index[random.randint(0, len(connected_index) - 1)]
        unit_map[connections[0], connections[1]] = 1
        count += 1
        
    weighted_matrix = position_weighting(unit_map)
    
    ones_floaters = np.where(weighted_matrix == 1)
    ones_index = list(zip(ones_floaters[0], ones_floaters[1]))
    
    connected_points = np.where(weighted_matrix >= 2)
    connected_index = list(zip(connected_points[0], connected_points[1]))
        
plt.imshow(unit_map)
plt.show()
plt.imshow(weighted_matrix)
plt.show()

np.save('material_map', unit_map)

print(np.mean(unit_map))


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

#close the Gmsh API    
gmsh.finalize()



