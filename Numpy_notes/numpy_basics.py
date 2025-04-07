#                USE OF NUMPY ARRAY
'''
 Used to do mathametical operation for ML, DS

'''

#             NP ARRAY OPERATIONS
'''
import numpy as np

a=[[1,3,1],[4,6,1]]
nump = np.array(a)
print(nump * 2) # in normal array the array will printed in 2 times
               # instead of mul each element in array.
               # this is how it is used to do mathematical operations
               # on array

print(nump > 5)

'''
#                RAND INT
'''
from numpy import random
print(random.randint(10,100)) # returns single random value

y = np.random.randint(100, size=(3)) # creates 3 random values
'''

#                  NUMPY ARRAY ADVANTAGE
'''
Numpy array shares single space instead of allocation new space
for new processed array.

Standard Python objects take up more memory than NumPy objects 
operations on NumPy objects complete very quickly compared to 
comparable objects in standard Python.

import numpy as np

nump1= np.array([1,2,4,5,6,6])
nump2 = nump1[::2]

print(nump2)
print(np.shares_memory(nump1, nump2)) # True

OUTPUT:

[1 4 6]
True
'''

#                 SLICING 2D NUMPY ARRAY
'''
import numpy as np

nump1= np.array([[1,2,4],[5,6,6]])
print(nump1[0, 1]) # 2
'''

#                 MASKING FOR EVEN NUMBER
'''

Conditional based filtering the array elements

import numpy as np

nump1= np.array([[1,2,4],[5,6,6]])
mask = nump1 %2 == 0 # returns of list of bool
print(nump1[mask])

OUTPUT:
[2 4 6 6]

'''

#                     MATRIX
'''

It is like array but it can have only 2D array while array can 
have nD array. It is deprecated, It doesn't perform element wise
calculation.

import numpy as np

nump1= np.matrix([[1,2,4],[5,6,6]])
remainder = nump1 %2
print(remainder)

numpy array is the good one for tensor like matrix
'''

#                   BOARD CASTING
'''
Helps to match both array's shape to perform math operations

import numpy as np
nump1= np.array([[1,2,3],[4,5,6]])
nump2= np.array([[7],[3],[5]])
nump2 = nump2[:, np.newaxis] # 
print(nump1+nump2) # this addition will gives error without
                # the above newaxis
'''

#               RESHAPE
'''

Converts rows to column, and columns to rows

import numpy as np

nump1= np.array([[1,2,3],[4,5,6]])
nump2= np.array([[7],[3],[5]])

print(np.reshape(nump2,(1,3)))

OUTPUT:
[[7 3 5]]
'''

#                INSERT, APPEND, DELETE Numpy array element
'''

import numpy as np

nump1= np.array([[1,2,3],[4,5,6]])
nump2= np.array([[7],[3],[5]])
nump3 = np.append(nump1, [[1,4,5]], 0) # appends 1D array to last as row
                                       # 0 indicates values appends as row
                                       # to append as column 1 is used.

nump4 = np.insert(nump1,[3], [[1],[4]], 1) # inserts 1 and 4 in as new column
                                           # specified column shape should
                                           # equal to the row in specified
                                           # array

nump5 = np.delete(nump2, [2], 0) # deletes 2nd element in row
print(nump5)
'''

#              ARANGE, ZEROS, DIAG
'''
import numpy as np

print(np.arange(start=1, stop=10, step=2))# 1D array

nump1= np.arange(1,7,3) # creates 1D array from 1 to 10 values
nump1 = np.zeros(4) # create zeros 1D array

nump2 = np.diag((1,10, 11)) # creates diagnal 2D matrix
# where:
# 1 is placed at position (0,0)
# 10 is placed at position (1,1)
#other are 0
print(nump2)

OUTPUT:

[[ 1  0  0]
 [ 0 10  0]
 [ 0  0 11]]

nump3 =  np.random.random(10) # randomly generates 10 floating
                              # numbers while arange creates 
                              # whole number

nump4 = nump3.mean() # return mean
print(nump4)
'''

#                       FILL METHOD
'''
import numpy as np

np.full(shape=(4,4), fill_value=[i for i in range(2*2)])
OUTPUT:
[[0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]
 [0 1 2 3]]
'''

#             LINSPACE
'''
Generated evenly seperated 1D 2D array

import numpy as np
print(np.linspace(start=0,stop=0.6,num=3))

OUTPUT:
[0.  0.3 0.6]

import numpy as np
print(np.linspace(start=[1,2,3],stop=[4,5,6],num=3))

OUTPUT:
[[1.  2.  3. ]
 [2.5 3.5 4.5]
 [4.  5.  6. ]]
'''
import numpy as np
print(np.linspace(start=[1,2,3],stop=[4,5,6],num=3))