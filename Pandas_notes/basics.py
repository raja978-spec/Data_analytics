#                  INTRO
'''
 * Pandas suits for tabular data with heterogeneously typed columns
   are in SQL table or Excel sheet

 * Ordered and unordered time series data.

 * Series - 1D, DataFrame - 2D Multi dimensional data type

 * Helps in reshaping data 

 * Pandas build in top of numpy for data analysis, it 
   is an enchaned version of numpy.
'''

#     BUILDING BLOCKS OF PANDAS
'''
 In pandas table the first column always will be index that
 represents the location of each row data.

 Columns are denoted ny axis = 0 or 1, combination of row and
 column constrctions DataFrame, Series will be one single column
 or collection of columns.
'''

#  READING DOCUMANTATION OF PANDAS IN JUPYTER
'''
 pd? - help

 pd?? - more help

'''

#               RETERIVE DATA WITH N
'''
import pandas as pd
books_df = pd.read_csv('Pandas_notes/dataset/titanic.csv')
print(books_df.head()) # defaults retries first 5 rows
print(books_df.head(n=10)) # first 10 rows
print(books_df.tail(n=10)) # last 10 rows
print(books_df.columns) # column names available in dataset
print(books_df.index) # index names available in dataset

# We will got column names and index datatype in dataset
# no of no null values
print(books_df.info()) # info about dataset

# Returns mean, median, std, min, max of all columns
# What will happen if a column has string?
# Answer is NaN
print(books_df.describe())

print(books_df.shape) # gives me no of rows and columns

'''

#               EXTRACT SINGLE COLUMN
'''
import pandas as pd
books_df = pd.read_csv('Pandas_notes/dataset/titanic.csv')
d = books_df['survived']
print(type(d))
'''

#                 SERIES
'''
import pandas as pd

# Series(data, index)
# data should be one dimensional array
books_s = pd.Series([1,2,34])
#print(books_s)

# If we give dict ad data the key will become index 
books_s = pd.Series({'book1': 'A', 'book2': 'B', 'book3': 'C'})
#print(books_s)

#  Defining only on data for series
# Here the value can be constrcuted for 4 times
books_s = pd.Series(4, index=[1,2,3,4])
print(books_s)
# OUTPUT:
# 1    4
# 2    4
# 3    4
# 4    4
# dtype: int64
# <class 'pandas.core.series.Series'>

# We can grab only the wanted series value
# by specifying the index which we want
books_s = pd.Series({'book1': 'A', 'book1': 'B', 'book3': 'C'}, index=['book1'])
#print(books_s)
# The above code will shows only the last book1 data
# OUPUT: book1    B
#        dtype: object


# Reading csv files as series
books_s = pd.read_csv('Pandas_notes/dataset/Marks_list.csv')
print(type(books_s['Student_Name'])) # returns type of each object
#OUTPUT: <class 'pandas.core.series.Series'>

                 TO LIST

 converts series to normal python list

 s4 = pd.Series([i for i in range(10)], index=keys)
 print(s4.shape, s4.size)
 print(s4[4::3], s4[::-1], s4[::2])
 L= s4.tolist()

               CONDITIONAL SERIES SLICING

s5=pd.Series([25,50,100,200,300,500,600,700])
s = s5[s5 >= 100]
print(s)


'''
#                    DF

'''
import pandas as pd

# Dataframe(data, index, columns)
# DF can be aligned series object
# Each and every dimension are considered as row
# The data can be dict, array, series, constants list
book_df = pd.DataFrame(data=[[1,23,4],["hello", 'ahi']],index=[1,2], columns=['coln1', 'col2','col3'])
#print(book_df)

# Creating multi dict dataframe
states ={
    'kar':'ben'
}

state_lab = {
    'kar': 'kanada'
}

states_CL = pd.DataFrame({'cp':states, 'lan':state_lab})
#print(states_CL)
# here the key of each dict same the values of each sep
# dict becomes values and the key becomes index
# OUTPUT:
#       cp     lan
# kar  ben  kanada

states_CL = pd.DataFrame({'cp':pd.Series([1,2,4]), 'lan':[5,6,7]}, index=['a','b','c'])
#print(states_CL)
# OUTPUT:
#    cp  lan
# a NaN    5
# b NaN    6
# c NaN    7

# Here pd.Series([1,2,4]) will have the default index [0,1,2] 
# and [5,6,7] will have the index we specified
# To fix this we have to specify the index inside the series
states_CL = pd.DataFrame(
    {
    'cp':pd.Series([1,2,4], index=['a','b','c']), 
    'lan':[5,6,7]
    }
    )
#print(states_CL)

                    APPLY, DROPNA
exam_data = {
    'name':['a','b','c','d','e','f','g'],
    'score':[10,20,np.nan,40,50,np.nan,70],
    'attempts':[1,3,4,5,6,7,5],
    'qualify':['s','n','s','s','s','n','n']
}
labels = ['a','b','c','b','e','f','g']
df =pd.DataFrame(exam_data,index=labels)

def findScore(s):
    if s > 22:
        return s
s7=df['score'].apply(findScore)
s8=s7.dropna() #drops all null value and
               # returns the non-null values

                
                 CONDITIONAL DF SLICILNG

exam_data = {
    'name':['a','b','c','d','e','f','g'],
    'score':[10,20,np.nan,40,50,np.nan,70],
    'attempts':[1,3,4,5,6,7,5],
    'qualify':['s','n','s','s','s','n','n']
}
labels = ['a','b','c','b','e','f','g']
df =pd.DataFrame(exam_data,index=labels)

# All conditional operator can be worked here
s7=df[df['score']>22]
print(s7)

OR

s7=df.loc[df['score']>22]
print(s7)

Both of this gives sam output

OUTPUT:
  name  score  attempts qualify
b    d   40.0         5       s
e    e   50.0         6       s
g    g   70.0         5       n

                
                   SLICING DF WITH NOTNULL AND ISNULL
exam_data = {
    'name':['a','b','c','d','e','f','g'],
    'score':[10,20,np.nan,40,50,np.nan,70],
    'attempts':[1,3,4,5,6,7,5],
    'qualify':['s','n','s','s','s','n','n']
}
labels = ['a','b','c','b','e','f','g']
df =pd.DataFrame(exam_data,index=labels)
s7=df.loc[df['score'].notnull()]
s8 = df.loc[df['score'].isnull()]
print(s7,s8)

OUTPUT:

 name  score  attempts qualify
a    a   10.0         1       s
b    b   20.0         3       n
b    d   40.0         5       s
e    e   50.0         6       s
g    g   70.0         5       n   

    name  score  attempts qualify
c    c    NaN         4       s
f    f    NaN         7       n


                  EXTRACTING MORE THAN ONE COLUMN

exam_data = {
    'name':['a','b','c','d','e','f','g'],
    'score':[10,20,np.nan,40,50,np.nan,70],
    'attempts':[1,3,4,5,6,7,5],
    'qualify':['s','n','s','s','s','n','n']
}
labels = ['a','b','c','b','e','f','g']
df =pd.DataFrame(exam_data,index=labels)

# We we want to extract more than one column than 
# we should pass the columns as list to the
# array selector([])
s7=df[['name','attempts']]
print(s7)

OUTPUT:

     name  attempts
a    a         1
b    b         3
c    c         4
b    d         5
e    e         6
f    f         7
g    g         5


                   VALUE_COUNTS

# counts no of occurence of each value in this column
s7=df['attempts'].value_counts()
 
                      T

import pandas as pd

state_cap = pd.Series({'kar':1,'andra':2,'tn':3})
state_lan = pd.Series({'kar':1,'andra':2,'tz':3})

data = pd.DataFrame({'capitals':state_cap,'lang':state_lan})
print(data)
print(data.T) # changes row wise data to column wise

OUTPUT:
       capitals  lang
andra       2.0   2.0
kar         1.0   1.0
tn          3.0   NaN
tz          NaN   3.0
          andra  kar   tn   tz
capitals    2.0  1.0  3.0  NaN
lang        2.0  1.0  NaN  3.0

      
                      VALUES
            
import pandas as pd

state_cap = pd.Series({'kar':1,'andra':2,'tn':3})
state_lan = pd.Series({'kar':1,'andra':2,'tn':3})

data = pd.DataFrame({'capitals':state_cap,'lang':state_lan})
print(data)
print(data.values) # returns all row values as list

OUTPUT:
       capitals  lang
kar           1     1
andra         2     2
tn            3     3
[[1 1]
 [2 2]
 [3 3]]


'''
#                   INDEX


'''
import pandas as pd

# Index is also like a data structure that can be ordered like set
# Responsible for holding axis and other meta data
# by default any array or other sequence used to constract index

# Index(data, dtype, name)
# data can be list, array, series
# name is the name of the index

# Index 
# It is a immutable array, may contain repeat values
list = [1,2,3,4,5]
ind=pd.Index(list)
#print(ind)
# OUPUT: Index([1, 2, 3, 4, 5], dtype='int64', name='integers')

#ind[1] = '2' # throws error

#print(ind[::2])
# OUPUT: Index([1, 3, 5], dtype='int64', name='integers')

#print(ind.size) # num of rows
# OUPUT: 5
# All the series, DF, index has the prop that is shape, ndim, dtype
# and size

# We can perform set operations on index
list2= [1,5,7,4,78]
ind2=pd.Index(list2)
print(ind.intersection(ind2)) # intersection
print(ind.union(ind2)) # union
print(ind.difference(ind2)) # difference
print(ind2.difference(ind))
print(ind.symmetric_difference(ind2)) # symmetric difference

OUTPUT:
Index([1, 4, 5], dtype='int64')
Index([1, 2, 3, 4, 5, 7, 78], dtype='int64')
Index([2, 3], dtype='int64')
Index([7, 78], dtype='int64')
Index([2, 3, 7, 78], dtype='int64')
'''

#        TO_NUMERIC METHOD
'''
print(pd.to_numeric(df['score']))
'''

#                    DATA INDEXING SELECTION
'''
 It means the accessing and modifying the values 

 Series Indexing

import pandas as pd
data = pd.Series([10,20,20], index=['A','B','C'])
print(data['A'])  #Accesing the item with index
print(data.keys())
print(list(data.items())) # returns iterable value pair index,value
data['A'] = 45
print(data)

OUTPUT:

10
Index(['A', 'B', 'C'], dtype='object')
[('A', 10), ('B', 20), ('C', 20)]
A    45
B    20
C    20
dtype: int64


OUTPUT:
A    10
B    20
C    30
D    40
dtype: int64
A    10
B    20
C    30
dtype: int64


import pandas as pd
data = pd.Series([10,20,30,40], index=['A','B','C','D'])
print(data['A':'D']) # Explicit index included D
print(data[0:3]) # implicit index excludes 3th index that is D
print(data[(data>20) & (data<40)]) # conditional indexing
print(data[['A','D']])


OUTPUT:

A    10
B    20
C    30
D    40
dtype: int64
A    10
B    20
C    30
dtype: int64
C    30
dtype: int64
A    10
D    40
dtype: int64
'''

#          LOC, ILOC
'''

             SERIES

import pandas as pd
data = pd.Series([10,20,30,40], index=[1,2,3,4])
print(data[1]) # uses explicit index

# But when we do slicing program get consfuse and
# it uses default implicit index for slice
print(data[1:3])

OUTPUT:
10
2    20
3    30

Because of this confusion loc comes in, it uses
explicit index for slice


import pandas as pd
data = pd.Series([10,20,30,40], index=[1,2,3,4])
print(data.loc[1]) # uses explicit index

# But when we do slicing program get consfuse and
# it uses default implicit index for slice
print(data.loc[1:3])


OUTPUT:

10
1    10
2    20
3    30
dtype: int64


What if we want to slice using implicit on both
single item, here iloc is used, it will throw
IndexError if the item is not found


import pandas as pd
data = pd.Series([10,20,30,40], index=[1,2,3,4])
 # uses implicit index
print(data.iloc[1])
print(data[1])
print(data.iloc[1:3])

OUTPUT:
20
10
2    20
3    30
dtype: int64


                    DF IMPLICIT AND EXPLICIT SLICING  

import pandas as pd

state_cap = pd.Series({'kar':1,'andra':2,'tn':3})
state_lan = pd.Series({'kar':1,'andra':2,'tn':3})

#data = pd.DataFrame({'capitals':state_cap,'lang':state_lan})
data = pd.DataFrame({'col1':[1,2,3],'col2':[4,5,6], 'col3':[7,8,9]}, index=['row1','row2', 'row3'])
print(data)

NOTE: implict slicing for Series works on column
      But here it applies on row

print(data[0:2]) # slices the data row-wise, uses implict so the 
                # last index will be excluded.
print(data['row1':'row3']) # this uses explicit index to it includes
                           # last row

OUTPUT:
      col1  col2  col3
row1     1     4     7
row2     2     5     8
row3     3     6     9

      col1  col2  col3
row1     1     4     7
row2     2     5     8

      col1  col2  col3
row1     1     4     7
row2     2     5     8
row3     3     6     9


                 DF LOC, ILOC, IX

 1. Loc - in df loc it access group of rows and columns b labels
          it single label, array of labels

 2. iloc - purely index based 

 3. ix - combination of implicit and explicit accessing. but it is
         deprecated


EXAMPLE FOR LOC:

import pandas as pd
data = pd.DataFrame({'col1':[1,2,3],'col2':[4,5,6], 'col3':[7,8,9]}, index=[1,2,3])
print(data[1:3])
print(data.loc[1:3])


OUTPUT:

   col1  col2  col3
2     2     5     8
3     3     6     9

   col1  col2  col3
1     1     4     7
2     2     5     8
3     3     6     9


import pandas as pd
data = pd.DataFrame(
    {'col1':[1,2,3],'col2':[4,5,6], 
    'col3':[7,8,9]}, 
    index=['row1','row2','row3'])
print(data)
print(data.loc['row2':,'col2':])

OUTPUT:

      col1  col2  col3
row1     1     4     7
row2     2     5     8
row3     3     6     9

      col2  col3
row2     5     8
row3     6     9
 
            
             EXAMPLE FOR ILOC:

import pandas as pd
data = pd.DataFrame({'col1':[1,2,3],'col2':[4,5,6], 'col3':[7,8,9]}, index=[1,2,3])
print(data)
print(data.iloc[:3,0:2]) # iloc[row_index,col_index]


OUTPUT:

   col1  col2  col3
1     1     4     7
2     2     5     8
3     3     6     9

   col1  col2
1     1     4
2     2     5
3     3     6


               EXAMPLE FOR IX 

import pandas as pd
data = pd.DataFrame(
    {'col1':[1,2,3],'col2':[4,5,6], 
    'col3':[7,8,9]}, 
    index=['row1','row2','row3'])
print(data)
print(data.ix['row2':,2:])


                CONDITIONAL LOC INDEXING


import pandas as pd
data = pd.DataFrame(
    {'col1':[1,2,3],'col2':[4,5,6], 
    'col3':[7,8,9]}, 
    index=['row1','row2','row3'])
print(data)

# In the below code, the condition
# data['col1']>1 is applied on col1
# it selects the condition applied data
# and selects the col3 with this

print(data.loc[data['col1']>1,['col3','col1']])

OUTPUT:

      col1  col2  col3
row1     1     4     7
row2     2     5     8
row3     3     6     9

# we have not specify condition for col3, all the
# values of cal3 will not be selected it will does 
# reductions on dim to the condition applied
# for col1

      col3  col1
row2     8     2
row3     9     3


                    #  MODIFYING VALUES WITH ILOC, LOC

import pandas as pd
data = pd.DataFrame(
    {'col1':[1,2,3],'col2':[4,5,6], 
    'col3':[7,8,9]}, 
    index=['row1','row2','row3'])
print(data)
print(data.iloc[:1,:1])
print(data.loc['row2':'row2','col2':'col2'])
data.iloc[:1,:1] = 0
data.loc['row2':'row2','col2':'col2'] = 0
print(data)
print(data.loc[ data['col1']>1, ['col2','col1'] ])
data.loc[data['col1']>1,['col2','col1']] = 0
print(data)

OUTPUT:
      col1  col2  col3
row1     1     4     7
row2     2     5     8
row3     3     6     9

      col1
row1     1

      col2
row2     5

      col1  col2  col3
row1     0     4     7
row2     2     0     8
row3     3     6     9

      col2  col1
row2     0     2
row3     6     3


      col1  col2  col3
row1     0     4     7
row2     0     0     8
row3     0     0     9
'''

import pandas as pd
data = pd.DataFrame(
    {'col1':[1,2,3],'col2':[4,5,6], 
    'col3':[7,8,9]}, 
    index=['row1','row2','row3'])
print(data)
print(data.iloc[:1,:1])
print(data.loc['row2':'row2','col2':'col2'])
data.iloc[:1,:1] = 0
data.loc['row2':'row2','col2':'col2'] = 0
print(data)
print(data.loc[ data['col1']>1, ['col2','col1'] ])
data.loc[data['col1']>1,['col2','col1']] = 0
print(data)

#     SERIES DICT INDEXING
'''
import pandas as pd

state_cap = pd.Series({'kar':1,'andra':2,'tn':3})
state_lan = pd.Series({'kar':1,'andra':2,'tz':3})

# both of the column will uses same key which is specified
# in the both above dict
data = pd.DataFrame({'capitals':state_cap,'lang':state_lan})
data['ner'] = data['capitals']+ data['lang']
print(data)

OUTPUT:
       capitals  lang  ner
andra       2.0   2.0  4.0
kar         1.0   1.0  2.0
tn          3.0   NaN  NaN
tz          NaN   3.0  NaN
'''

