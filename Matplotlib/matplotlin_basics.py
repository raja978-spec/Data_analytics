#                     WHAT IS MATPLOTLIB
"""
It is a basics low level visualization library, because it needs more
effort to create a good visual chart, since it has alot of options to
edit each part of the chart. While seaborn has default theme for it's
chart.
"""

#                     PLOT
"""

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

"""

#                   ADD STYLES TO THE PLOT CHART
"""

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


"""

#                  DEFAULT VALUE OF X
"""
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

"""

#                    MARKER, SHORT HAND ARGUMENT
"""
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


"""

#                   LINESTYLE ls, color c
"""
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

"""


#          DRAW TWO OR MORE PLOT IN SAME GRAPH
"""
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

"""

#    XLABEL, YLABEL, TITLE, Font_dict attribute, TITLE alignment with loc
"""

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

"""

#            ADDING GRID LINE TO PLOT
"""
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

"""


#                   SUBPLOT WITH SUPTITLE

"""
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
 
"""

#               SUBPLOTS AND FIGSIZE

"""

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


"""

#               SUBPLOT WITH SINGLE ARGUMENT
"""
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
"""

#         SCATTER PLOT
"""
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


     DIFFERENCE BETWEEN LINE AND SCATTER PLOT
     
Line is of continous values to show trend over period. 

So why we can't use line instead of scater?

EX:
    
import matplotlib.pyplot as plt

students = ["A", "B", "C", "D", "E"]
scores = [85, 90, 78, 88, 92]

plt.plot(students, scores, marker="o")  # Line plot
plt.title("Exam Scores (Misleading Line Plot)")
plt.xlabel("Students")
plt.ylabel("Scores")
plt.show()

👉 Here, connecting A → B → C → D → E suggests there is a trend or continuity between students, which doesn’t exist (students are just categories).
👉 The order of categories must have meaning for a line plot. so we can move scatter



           DIFFERENCE FOR BAR AND SCATTER

Bar chart → Better for category vs single value (clear comparisons).

Scatter plot → Better when you have two numeric variables and want to study the 
relationship (not just compare categories).


EX:
    
import matplotlib.pyplot as plt

hours_studied = [1, 2, 3, 4, 5, 6, 7, 8]
exam_scores   = [35, 40, 50, 55, 60, 70, 78, 85]

plt.scatter(hours_studied, exam_scores)
plt.title("Scatter Plot - Hours Studied vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.show()

👉 Why we can't use bar Here:

Both axes are numeric.

Scatter shows the relationship (positive correlation: more hours → higher score).

A bar chart would just put bars side by side and lose the pattern/relationship.

🔑 Why scatter is better here:

You can see the trend (upward slope).

You can spot outliers (like a student who studied a lot but scored low).

You can later add a regression line to quantify the relationship.


                        EXAMPLE FOR OUTLIER:

                            
import matplotlib.pyplot as plt

# Hours studied (x) vs exam scores (y)
hours_studied = [1, 2, 3, 4, 5, 6, 7, 8, 9]
exam_scores   = [30, 40, 50, 55, 65, 70, 78, 82, 20]  # last point is an outlier

plt.scatter(hours_studied, exam_scores, color="blue")
plt.title("Scatter Plot - Hours Studied vs Exam Score (with Outlier)")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")

# Highlight outlier
plt.scatter(9, 20, color="red", s=100, label="Outlier")
plt.legend()

plt.show()


👉 Here:

Most points show a positive relationship (more hours studied → higher score).

The last student studied 9 hours but scored only 20 → that’s an outlier, 
and scatter plot makes it immediately visible.

"""


#        SPECIFYING COLORS FOR EACH DOT IN SCATTER PLOT
"""
import matplotlib.pyplot as plt 
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
colors = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])

plt.scatter(x, y, c=colors)
plt.show()

# length of the colour array should be same as the axis data length
"""

#           COLOR MAPS(cmap) ATTRIBUTE IN SCATTER PLOT
"""
The Matplotlib module has a number of available colormaps.

A colormap is like a list of colors, where each color has a 
value that ranges from 0 to 100.

This colormap is called 'viridis' and as you can see it ranges 
from 0, which is a purple color, up to 100, which is a yellow color.
"""

#              CHANGING SIZE OF EACH POINT IN SCATTER PLOT
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
size = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])

plt.scatter(x, y,s=size)

plt.show()
"""

#       MAKE TRANSPARENCY EFFECT TO EACH DOTS WITH ALPHA ATTRIBUTE
"""
Alpha

You can adjust the transparency of the dots with the alpha argument.

import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
t = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])

plt.scatter(x, y,alpha=0.5)

plt.show()
"""

#        COMBINING COLOR MAP, SIZE, ALPHA
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.random.randint(100, size=(10))
y = np.random.randint(100, size=(10))
colors = np.random.randint(100, size=(10))
sizes = 10*np.random.randint(100, size=(10)) # Increasing each size * 10

plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='nipy_spectral')

plt.colorbar()

plt.show()
"""

#                 BAR CHART
#           This will not allows all the short hand prop of plot
"""
import matplotlib.pyplot as plt
import numpy as np

x = ['Apple','Orange','Mango']
y = np.random.randint(100, size=(3))
plt.bar(x,y,width=0.2)
plt.show()


WHEN TO USE BAR PLOT

* You want to compare values across categories 
  (e.g., sales by product, marks by student).

* The data is discrete / categorical (not continuous 
                                    like time series).

"""

#          HORIZONDAL BAR CHART
"""
import matplotlib.pyplot as plt
import numpy as np

x = ['Apple','Orange','Mango']
y = np.random.randint(100, size=(3))
plt.barh(x,y, color='r')
plt.show()

"""

#                HISTOGRAM
"""

Helps to check the frequency or range of particular number for large numeric 
dataset, so it Shows the distribution of continuous data 
by grouping values into bins


EX: how many peoples have height from 140, 145

import matplotlib.pyplot as plt
import numpy as np

x=[145,140,155,140, 147, 151, 148, 148]
plt.hist(x)
plt.show()


                  BINS in HISTOGRAM

Helps to control the frequency or range of the dataset

Bins are automatically using numpy auto statergy, it applies
different different methods to find the best bin


EX:
    
import matplotlib.pyplot as plt

data = [1,2,2,3,3,3,4,4,5,6,7,8,9,10]
plt.hist(data)   # bins chosen automatically
plt.show()


EXPLANATION ABOVE CHART:

1. What bins mean

The x-axis is split into intervals (bins).

Each bar shows how many data points fall into that interval.

For example:

Bin 1–2 → count how many values are in this range.

Bin 2–3 → count values in this range.

…and so on.

2. Why the bars have different heights

The height of each bar = frequency (number of values inside that bin).

Example: If you had data like [1, 2, 2, 3, 3, 3, 4, 4, 10]:

3 occurs 3 times → so the bar at bin “around 3” is tallest.

10 occurs once → so the bar at bin “around 10” is small.

3. How bins are chosen automatically

By default, Matplotlib uses NumPy’s “auto” bin rule → it tries to find a “nice” number of bins depending on your data spread and size.

In your case, it split into a few wide bins (like 1–2, 2–3, 3–4, 4–5, and a big one covering 5–10).
    

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


              SCENARIOS WHERE WE CAN USE HISTOGRAM
              
              
AGE DISTRIBUTION

👉 This shows how many people fall into each age group (like 18–20, 21–23, etc.).


import matplotlib.pyplot as plt

ages = [18, 19, 20, 21, 21, 22, 22, 22, 23, 24, 25, 25, 26, 27, 30]

plt.hist(ages, bins=6, edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age Range")
plt.ylabel("Number of People")
plt.show()


EXAM SCORE RANGE


  CHANGING DEFAULT Y-AXIS VALUE(FREQUENCY) IN HISTOGRAM
  
  
1. Density - used of probability distribution(0-1)

import matplotlib.pyplot as plt

data = [1,2,2,3,3,3,4,4,5,6,7,8,9,10]
plt.hist(data, bins=5, density=True, edgecolor="black")
plt.title("Density (Probability)")
plt.show()


2. custom value with Weight


import matplotlib.pyplot as plt
import numpy as np

data = [1,2,2,3,3,3,4,4,5,6,7,8,9,10]

weights = np.ones_like(data) / len(data)   # normalize to 1
print(weights, len(data), len(weights))
plt.hist(data, bins=5, weights=weights, edgecolor="black")
plt.title("Histogram - Percentage")
plt.ylabel("Percentage")
plt.show()



[0.07142857 0.07142857 0.07142857 0.07142857 0.07142857 0.07142857
 0.07142857 0.07142857 0.07142857 0.07142857 0.07142857 0.07142857
 0.07142857 0.07142857] 14 14

| Bin           | Count | Weighted sum         |
| ------------- | ----- | -------------------- |
| 1 `[1,2.8)`   | 3     | 3 * 0.07143 ≈ 0.214 |
| 2 `[2.8,4.6)` | 5     | 5 * 0.07143 ≈ 0.357 |
| 3 `[4.6,6.4)` | 2     | 2 * 0.07143 ≈ 0.143 |
| 4 `[6.4,8.2)` | 2     | 2 * 0.07143 ≈ 0.143 |
| 5 `[8.2,10]`  | 2     | 2 * 0.07143 ≈ 0.143 |

"""



#                FIGURE
"""
import matplotlib.pyplot as plt
import numpy as np

x=np.random.randint(1000, size=10)
y=np.random.randint(1000, size=10)+2

plt.figure(figsize=(10,3))
plt.hist([x, y],
         edgecolor='black', stacked=True)
plt.legend(['Group1', 'Group2'])
plt.show()

"""
#        XTICKS, ROTATION, TEXT
"""
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

"""

#     XTICKS WITH ha attribute
"""
Helps to control the horizontal space of the ticks

x=list(all_equipment_mean.loc['mean',:].index)
y=list(all_equipment_mean.loc['mean',:].values)
plt.plot(x,y,c='gold',marker='*',markersize=10,
         mfc='red', mec='black')
plt.xticks(rotation=45, ha='right') 
plt.title('Fourth Movement business decision')
plt.show()

"""

#   CLA and CLOSE METHOD in PLT
"""
Cla - clear current axis, without this chart data might get 
      overlapped to next charts, so instead of showing single
      analysis on each function first function's
      chart data getting appended to second, and first, second
      charts getting appended to third one

Close - closes the current chart process

EX:

import matplotlib.pyplot as plt

def generate_year_sales_report():
    plt.clf()  # Clear the current figure
    # or use plt.figure().clf() if you're using figure object

    # Your plotting code here
    plt.bar(['2024', '2025'], [100, 200])
    plt.title("Yearly Sales")
    plt.savefig("year_sales.jpg")
    plt.close()  # Important: closes the figure and frees memory

def generate_month_sales_report():
    plt.clf()  # Clear the current figure
    # or use plt.figure().clf() if you're using figure object

    # Your plotting code here
    plt.bar(['Jan', 'Feb'], [100, 200])
    plt.title("Month Sales")
    plt.savefig("month_sales.jpg")
    plt.close()  # Important: closes the figure and frees memory

def generate_week_sales_report():
    plt.clf()  # Clear the current figure
    # or use plt.figure().clf() if you're using figure object

    # Your plotting code here
    plt.bar(['1', '2'], [100, 200])
    plt.title("Week Sales")
    plt.savefig("week_sales.jpg")
    plt.close()  # Important: closes the figure and frees memory

generate_year_sales_report()
generate_month_sales_report()
generate_week_sales_report()

"""


#    WHEN TO USE LINE PLOT
'''

Use line plot for continuous data & trends.
And To compare trends of multiple datasets across the 
same continuous variable.

EX:

days = [1, 2, 3, 4, 5, 6, 7]
city1 = [30, 32, 31, 29, 28, 27, 26]
city2 = [22, 23, 21, 24, 25, 26, 27]

plt.plot(days, city1, label="City 1")
plt.plot(days, city2, label="City 2")
plt.legend()
plt.title("Temperature Trend")
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.show()

'''

# 




























