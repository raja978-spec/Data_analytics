#                     WHAT IS MATPLOTLIB
'''
It is a basics low level visualization library, because it needs more
effort to create a good visual chart, since it has alot of options to
edit each part of the chart. While seaborn has default theme for it's
chart.
''' 

#                     PLOT
'''

* Plot is used to draw point in a diagram
* Default it draws line between two points

SYNTAX: plot(x_axis, y_axis)

x and y can be a array containing points, if we need to plot
a line from 1 to 8 as x below is the example

EXAMPLE:

import matplotlib.pyplot as plt
import numpy as np

x_point= np.array([1,8])
y_point=np.array([10,50])

plt.plot(x_point, y_point)
plt.show()

'''

#                   ADD STYLES TO THE PLOT CHART
'''

In third argument it takes all the argument related to customize the
drawings on graph

SYNTAX:

plt.plot(x, y, 'marker linestyle color')

Marker → Shape of the point (e.g., 'o' for circle, 's' for square)

Line Style → Type of line connecting points (e.g., '-' for solid, '--' for dashed)

Color → Color of the line/marker (e.g., 'r' for red, 'g' for green)


Below are all marker styles, Line style, Colors available in matplotlib

Marker	Symbol
'.'	Small dot
','	Pixel
'x'	Cross
'+'	Plus
'*'	Star
's'	Square
'd'	Diamond
'v'	Triangle (down)

Style	Description
'-'	Solid line
'--'	Dashed line
'-.'	Dash-dot line
':'	Dotted line


Color Code	Color
'r'	        Red
'g'	        Green
'b'	        Blue
'y'	        Yellow
'k'	        Black
'm'	        Magenta
'c'	        Cyan

EXAMPLE:

import matplotlib.pyplot as plt
import numpy as np

x_point= np.array([1,8])
y_point=np.array([10,50])

#plt.plot(x_point, y_point, 'd-.r') 
plt.plot(x_point, y_point,'o-g',linewidth=6, 
         markersize=20, markerfacecolor='red', 
         markeredgecolor='black')
plt.show()


'''

#                  DEFAULT VALUE OF X
'''
If we do not specify the points on the x-axis, they will get the 
default values 0, 1, 2, 3 etc., depending on the length of the 
y-points.

import matplotlib.pyplot as plt
import numpy as np

x_point= np.array([1,8])
y_point=np.array([10,50])

plt.plot(y_point, 'sc', 
         linewidth=2, markerfacecolor='red', 
         markeredgecolor='gold',
         markersize=10)
plt.show()

'''

#                    MARKER, SHORT HAND ARGUMENT
'''
ms- is the short hand prop of marker size attribute
mfc - Marker face color
mec - marker edge color

LIST OF ALL AVAILABEL MARKERS

'o'	Circle	
'*'	Star	
'.'	Point	
','	Pixel	
'x'	X	
'X'	X (filled)	
'+'	Plus	
'P'	Plus (filled)	
's'	Square	
'D'	Diamond	
'd'	Diamond (thin)	
'p'	Pentagon	
'H'	Hexagon	
'h'	Hexagon	
'v'	Triangle Down	
'^'	Triangle Up	
'<'	Triangle Left	
'>'	Triangle Right	
'1'	Tri Down	
'2'	Tri Up	
'3'	Tri Left	
'4'	Tri Right	
'|'	Vline	
'_'	Hline


EX:

import matplotlib.pyplot as plt
import numpy as np

x_point= np.array([1,8])
y_point=np.array([10,50])

plt.plot(x_point, y_point,marker='*', ms='20')
plt.show()


'''

#                   LINESTYLE ls, color c
'''
ls- used to specify line style
color or c - used to specify the color of the line

 style	             Or
'solid' (default)	'-'	
'dotted'	        ':'	
'dashed'	        '--'	
'dashdot'	        '-.'	
'None'	           '' or ' '

import matplotlib.pyplot as plt
import numpy as np

x_point= np.array([1,8])
y_point=np.array([10,50])

plt.plot(x_point, y_point, ls='dotted')
plt.show()

'''


#          DRAW TWO OR MORE PLOT IN SAME GRAPH
'''
import matplotlib.pyplot as plt
import numpy as np

x_point= np.array([1,5])
y_point=np.array([10,50])
z_point=np.array([50,80])

# Below code plots three lines
plt.plot(x_point)
plt.plot(y_point)
plt.plot(z_point)

# Below code plots y, z and it draws line
# for both of them
#plt.plot(x_point, y_point, z_point)

plt.show()

'''

#    XLABEL, YLABEL, TITLE, Font_dict attribute, TITLE alignment with loc
'''

AXIS LABELS:

import matplotlib.pyplot as plt
import numpy as np

x_point= np.array([1,5])
y_point=np.array([10,50])
z_point=np.array([50,80])

plt.plot(x_point, y_point, z_point,
         lw=2, ls='dotted', marker='*',
         c='r', mfc='blue',
         mec='gold', ms='20')

plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()


TITLE WITH FONT DICT:

You can use the fontdict parameter in xlabel(), ylabel(), 
and title() to set font properties for the title and labels.

EX:

import matplotlib.pyplot as plt
import numpy as np

x_point= np.array([1,5])
y_point=np.array([10,50])
z_point=np.array([50,80])

plt.plot(x_point, y_point, z_point,
         lw=2, ls='dotted', marker='*',
         c='r', mfc='blue',
         mec='gold', ms='20')

font1 = {'family':'serif','color':'blue','size':18}
font2 = {'family':'serif','color':'darkred','size':15}

plt.title('Analysis of X and y', fontdict=font1)
plt.xlabel('X Axis', fontdict=font2)
plt.ylabel('Y Axis', fontdict=font2)
plt.show()



                  TITLE ALIGNMENT WITH LOC

import matplotlib.pyplot as plt
import numpy as np

x_point= np.array([1,5])
y_point=np.array([10,50])
z_point=np.array([50,80])

plt.plot(x_point, y_point, z_point,
         lw=2, ls='dotted', marker='*',
         c='r', mfc='blue',
         mec='gold', ms='20')


plt.title('Analysis of X and y', loc='left')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()

'''

#            ADDING GRID LINE TO PLOT
'''
With Pyplot, you can use the grid() function to add grid lines 
to the plot.

It will take all the short hand property of plot and one extra
property is axis which helps to show particular axis grid line, if
it is not specified both x and y grid line will be shown

import matplotlib.pyplot as plt
import numpy as np

x_point= np.array([1,5])
y_point=np.array([10,50])
z_point=np.array([50,80])

plt.plot(x_point, y_point, z_point,
         lw=2, ls='dotted', marker='*',
         c='r', mfc='blue',
         mec='gold', ms='20')

plt.grid(axis='x', c='yellow', lw='3')
plt.show()

'''


#                   SUBPLOT WITH SUPTITLE

'''
 In Matplotlib, subplots allow you to create multiple plots 
 (charts, graphs, or visualizations) within a single figure. 
 This is especially useful when you want to compare multiple datasets 
 or present related visualizations side-by-side.

 plt.subplot()
 This creates a single subplot within a grid.
 Syntax: plt.subplot(nrows, ncols, index)
 nrows: Number of rows in the grid.
 ncols: Number of columns in the grid.
 index: The position of the current subplot (1-based indexing).

 EXAMPLE:

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.title('Analysis 1')
plt.plot([1,5],[6,10], c='r')

plt.subplot(1,2,2)
plt.title('Analysis 2')
plt.plot([1,5],[6,10], c='g')
plt.suptitle('Whole analysis') # Adds title to whole chart

plt.tight_layout()
plt.show()
 
'''

#               SUBPLOTS AND FIGSIZE

'''

 The mani difference between subplot and subplots is in subplot
 we need to access each plot's index by specifying it's index, but in
 subplots each chart be can specified without passing third argument by 
 accessing axes returns by subplots.

 Tow items are returned by subplots():

 Figure: The overall container for all plots.
 Axes: The individual plots within the figure.
 Using subplots, you can arrange multiple axes in a grid-like 
 layout within a figure

 FigSize attribute helps to specify the layout of the whole chart

EX FOR 1D Chart:

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1,2,figsize=(10,6))

axes[0].plot([1,2,4],[5,6,7])
axes[0].set_title('Analysis 1')

axes[1].plot([1,2,4],[5,6,7])
axes[1].set_title('Analysis 2')

plt.show()


EXAMPLE FOR 2D CHARTS:

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2,2,figsize=(6,6))

axes[0,0].plot([1,2,4],[5,6,7])
axes[0,0].set_title('Analysis 1')

axes[0,1].plot([1,2,4],[5,6,7])
axes[0,1].set_title('Analysis 2')

axes[1,0].plot([1,2,4],[5,6,7])
axes[1,0].set_title('Analysis 3')

axes[1,1].plot([1,2,4],[5,6,7])
axes[1,1].set_title('Analysis 4')


plt.show()


'''

#               SUBPLOT WITH SINGLE ARGUMENT
'''
import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')

def plot():
    sns.lineplot(data=data, x='sepal_length', y='sepal_width')

with sns.axes_style('darkgrid'):
     plt.subplot(211) # Create one chart
     plot()

plt.subplot(212) # again created one char next to the previuos chart
plot()
plt.show()
'''

#         SCATTER PLOT
'''
Helps to compare two variable's relationship

The scatter() function plots one dot for each observation. 
It needs two arrays of the same length, one for the values 
of the x-axis, and one for values on the y-axis

EX:

import matplotlib.pyplot as plt 
import numpy as np

x_point=np.arange(1,10)
y_point=np.arange(10,1,-1)


plt.scatter(x_point, y_point, marker='o',c='r')
plt.show()

EXAMPLE FOR MULTIPLE SCATTER PLOT:

By default it will diffrentiat tow points

import matplotlib.pyplot as plt 
import numpy as np

# Day 1 analysis
x_point=np.arange(1,10)
y_point=np.arange(10,1,-1)
plt.scatter(x_point, y_point)

# Day 2 analysis
x_point=np.arange(2,20)
y_point=np.arange(20,2,-1)
plt.scatter(x_point, y_point)

plt.show()

'''

#        SPECIFYING COLORS FOR EACH DOT IN SCATTER PLOT
'''
import matplotlib.pyplot as plt 
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
colors = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])

plt.scatter(x, y, c=colors)
plt.show()

# length of the colour array should be same as the axis data length
'''

#           COLOR MAPS(cmap) ATTRIBUTE IN SCATTER PLOT
'''
The Matplotlib module has a number of available colormaps.

A colormap is like a list of colors, where each color has a 
value that ranges from 0 to 100.

This colormap is called 'viridis' and as you can see it ranges 
from 0, which is a purple color, up to 100, which is a yellow color.
'''

#              CHANGING SIZE OF EACH POINT IN SCATTER PLOT
'''
import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
size = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])

plt.scatter(x, y,s=size)

plt.show()
'''

#       MAKE TRANSPARENCY EFFECT TO EACH DOTS WITH ALPHA ATTRIBUTE
'''
Alpha

You can adjust the transparency of the dots with the alpha argument.

import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
t = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])

plt.scatter(x, y,alpha=0.5)

plt.show()
'''

#        COMBINING COLOR MAP, SIZE, ALPHA
'''
import matplotlib.pyplot as plt
import numpy as np

x = np.random.randint(100, size=(10))
y = np.random.randint(100, size=(10))
colors = np.random.randint(100, size=(10))
sizes = 10*np.random.randint(100, size=(10)) # Increasing each size * 10

plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='nipy_spectral')

plt.colorbar()

plt.show()
'''

#                 BAR CHART
#           This will not allows all the short hand prop of plot
'''
import matplotlib.pyplot as plt
import numpy as np

x = ['Apple','Orange','Mango']
y = np.random.randint(100, size=(3))
plt.bar(x,y,width=0.2)
plt.show()

'''

#          HORIZONDAL BAR CHART
'''
import matplotlib.pyplot as plt
import numpy as np

x = ['Apple','Orange','Mango']
y = np.random.randint(100, size=(3))
plt.barh(x,y, color='r')
plt.show()

'''

#                HISTOGRAM
'''

Helps to check the frequency or range of particular number for large numeric 
dataset

EX: how many peoples have height from 140, 145

import matplotlib.pyplot as plt
import numpy as np

x=[145,140,155,140, 147, 151, 148, 148]
plt.hist(x)
plt.show()


                  BINS in HISTOGRAM

Helps to control the frequency or range of the dataset

🔹 How bins Affects a Histogram

1. Fewer bins → Wider intervals → Less detail, but smoother visualization.

2. More bins → Narrower intervals → More detail, but can be too noisy.

If we give bin size greater the data size than nothing will happened.

EXAMPLE FOR 1:

import matplotlib.pyplot as plt
import numpy as np

x=[145,140,155,140, 147, 151, 148, 148]
plt.hist(x, bins=3)
plt.boxplot(x)
plt.show()

EXAMPLE 2:

import matplotlib.pyplot as plt
import numpy as np

x=np.random.randint(100, size=(20))
plt.hist(x, bins=10, color='red',histtype='step',
         orientation='horizontal')
plt.show()


                       ADD MULTIPLE HISTOGRAM

Overlap will occur when we add multi histo, to avoid this stacked are used

import matplotlib.pyplot as plt
import numpy as np

x=np.random.randint(1000, size=10)
y=np.random.randint(1000, size=10)+2

plt.hist([x, y],
         edgecolor='black', stacked=True)
plt.legend(['Group1', 'Group2'])
plt.show()

'''

#                FIGURE
'''
import matplotlib.pyplot as plt
import numpy as np

x=np.random.randint(1000, size=10)
y=np.random.randint(1000, size=10)+2

plt.figure(figsize=(10,3))
plt.hist([x, y],
         edgecolor='black', stacked=True)
plt.legend(['Group1', 'Group2'])
plt.show()

'''
#        XTICKS, ROTATION, TEXT
'''
plt.text(xaxis_location, y_axislocation, text_want_to_put)
#Helps to put text on graph

plt.xticks(rotation= -45)
#Helps to rotate the axis name

x=list(Machine_equipment_good_condition_mean_details.keys())
y=list(Machine_equipment_good_condition_mean_details.values())
print(y)
plt.plot(x,y,color='gold',marker='*',
         markerfacecolor='red',ms=10,
         mec='black')
for i in range(len(x)):
    plt.text(x[i], y[i], str(y[i]), ha='center', va='bottom')
plt.show()

'''

#     XTICKS WITH ha attribute
'''
Helps to control the horizontal space of the ticks

x=list(all_equipment_mean.loc['mean',:].index)
y=list(all_equipment_mean.loc['mean',:].values)
plt.plot(x,y,c='gold',marker='*',markersize=10,
         mfc='red', mec='black')
plt.xticks(rotation=45, ha='right') 
plt.title('Fourth Movement business decision')
plt.show()

'''