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

              
    GENERATING NORMALLY DISTRIBUTED RANDOM NUMPY ARRAY

import numpy as np
import matplotlib.pyplot as plt
# Generate 1000 random numbers from a normal distribution
data = np.random.normal(loc=50, scale=10, size=(100,100))
print(data)
# Plotting the histogram
plt.hist(data, bins=30, edgecolor='black')
plt.title('Normal Distribution (mean=50, std=10)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

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

#              ARANGE, ZEROS, DIAG, RANDOMS
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


                            RANDOMS

import numpy as np
import matplotlib.pyplot as plt

print(np.random.rand(3,2))# row column parameter, returns floating
                          # point values like 0.34234 in the give shape

print(np.random.randint(low=0, high=10, size=(100))) # return random list
                                                     # of int

plt.plot(np.random.randn(10)) # return combination of minus and plus floating
                              # values in given size
#plt.show()
print(np.random.randn(10))

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

#                         RANDOM SEED
'''
random.seed() method in Python is used to initialize the random 
number generator, ensuring the same random numbers on every run. 
By default, Python generates different numbers each time, but using 
.seed() allows result reproducibility.

Syntax
random.seed(a=None, version=2)

Parameters:

a (Optional): it’s the seed value (integer, float, str, bytes, or 
bytearray). If None, the system time is used.

version(Optional): defaults value is 2, using a more advanced seeding 
algorithm. version=1 uses an older method.
Return Type

random.seed() method does not return any value.

EX:
import numpy as np
np.random.seed(0) # we can any int value as seed
print(np.random.randn(1,2))

'''

#    ZERO LIKE
'''
Creates give df shape 0 numpy array with same
data type

import numpy as np 
a=np.array([[1,2],[3,4],[5,6],[7,8]],
dtype=np.float64
)
b=np.zeros_like(a)
print(b)

OUTPUT:
[[0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]]
'''

#   WHERE 
'''
Helps to replace values with some condition

And also it helps to access the numpy array elements
with index

import numpy as np 
a=np.array([[1,2],[3,4],[5,6],[7,8]],
dtype=np.float64
)
b=np.where(a%2==0,0,a)
print(b)

OUTPUT:
[[1. 0.]
 [3. 0.]
 [5. 0.]
 [7. 0.]]
'''

#   ACCESSING NUMPY ARRAY ELEMENT WITH CONDITION
'''
We can't specify conditions with and

import numpy as np 
a=np.array([[1,2],[3,4],[5,6],[7,8]],
dtype=np.float64
)
print(a[a>2])

OUTPUT:
[3. 4. 5. 6. 7. 8.]
'''

#          SORT IN NUMPY ROW AND COLUMNS
'''
import numpy as np 
a=np.array([[435,455],[4,53],[645,355],[535,8]],
dtype=np.float64
)
print(np.sort(a,axis=0)) # sorts columns
print(np.sort(a,axis=1))# sorts row

OUTPUT:
[[  4.   8.]
 [435.  53.]
 [535. 355.]
 [645. 455.]]

[[435. 455.]
 [  4.  53.]
 [355. 645.]
 [  8. 535.]]
'''

#       VSTACK
'''
Helps to add new arrays as rows to existing array

import numpy as np 
a=np.array([[435,455],[4,53],[645,355],[535,8]])
b=np.array([[5,5],[4,5],[6,35],[35,85]])
comb = np.vstack(a,b)
print(comb)

OUTPUT:
[[435 455]
 [  4  53]
 [645 355]
 [535   8]
 [  5   5]
 [  4   5]
 [  6  35]
 [ 35  85]]
'''

#   ACCESSING 2D ARRAY WITH ELEMENTS
'''
SYNTAX:
array_name[[start_row_no,end_row_no],:]

import numpy as np 
a=np.array([[435,45,5],[4,5,3],[645,35,5],[5,35,8]])
a[[0,-1],:] = a[[-1,0],:]
print(a)
'''

#      INVERSE MATRIX ON LINEAR ALGEBRA
'''
import numpy as np 
a=np.array([[435,45,5],[4,5,3],[645,35,5]])
print(np.linalg.inv(a))

OUTPUT:
[[-0.00222531 -0.00139082  0.00305981]
 [ 0.05326843 -0.02920723 -0.03574409]
 [-0.08581363  0.38386648  0.05549374]]
'''

#      REMOVING MISSING VALUES WITH ISNAN
'''
Returns bool for each element in numpy array

import numpy as np 
a=np.array([[435,45,5],[4,0,3],[0,35,5]])
print(np.isnan(a))

OUTPUT:
[[False False False]
 [False False False]
 [False False False]]
'''

#       FULL IN NP
'''
Helps to create same value array

import numpy as np
print(np.full(5,6)) 

OUTPUT: 
[6 6 6 6 6]
'''

#       NUMPY AS_TYPE 
'''
Helps to change the type of a an numpy array.

import numpy as np

a=np.array([1,2,4,4])
print(a.astype(float)) # Returns the converted data
#a.dtype=float
print(a)
'''

#           CONCATENATE IN NP
'''
Are helps to concatenate more than two numpy array 
Adds new array to the end of the original array

import numpy as np
a=np.array([[1,2,3],[5,6,7]])
b=np.array([[8,9,10],[11,12,13]])
c=np.concatenate((a,b), axis=0)

OUTPUT:
[[ 1  2  3]
 [ 5  6  7]
 [ 8  9 10]
 [11 12 13]]


d=np.concatenate((a,b), axis=1) # On this first row each concat as first array
print(d)

OUTPUT:
[[ 1  2  3  8  9 10]
 [ 5  6  7 11 12 13]]
'''     


#                HSTACK
'''
Converts data to horizontal, axis cannot be used here

'''
#               DSTACK
'''
Takes first row's first element of numpy array, axis cannot be used here

EX:
import numpy as np
a=[1,3,4]
b=[4,5,6]
c=np.vstack((a,b)) 
d=np.hstack((a,b))
e=np.dstack((a,b))
print(c)
print(d)
print(e)

OUTPUT:
[[1 3 4]
 [4 5 6]]
[1 3 4 4 5 6]
[[[1 4]
  [3 5]
  [4 6]]]
'''

#             SPLIT
'''
Splits the array in specified shape

import numpy as np
a= np.array([1,2,3,4,5,6])
b=np.array_split(a,3) # splits one array as 3 array
print(b)

OUTPUT:
[array([1, 2]), array([3, 4]), array([5, 6])]
'''

#         VIEW IN NUMPY
'''
Same as numpy array copy but changes made in view array
will changes the original array.

EX:
import numpy as np
a=np.array([[1,3,4],[4,5,6]])
b=a.view()
b[0,1]=42
print(a)

OUTPUT:
[[ 1 42  4]
 [ 4  5  6]]
'''

#            3D ARRAY
'''
axis-0: Operates along row
axis-1: Operated along column
axis-2: Operates along column depth
'''

#           NDITER
'''
nditer are helps iterate over n dim array
without this we need to give two for loop to
print the element in n dim arrya.

import numpy as np
a=np.array([[1,3,4],[4,5,6]])

for i in np.nditer(a):
    print(i)

OUTPUT:
1
3
4
4
5
6
'''
import numpy as np
a=[1,3,4]
b=[4,5,6]
c=np.vstack((a,b)) 
d=np.hstack((a,b))
e=np.dstack((a,b))
print(c)
print(d)
print(e)
