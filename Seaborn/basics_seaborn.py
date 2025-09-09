#          ABOUT SEABORN
'''
Seaborn has more advantages then matplotlib

1. Has built in theme to create good looking chart in minute
2. Pandas integration - we can plot dataframe
3. Syntax
4. And also it can be integrated with matplotlib
'''

#                 LINE PLOT
'''
SYNTAX:

lineplot(data=data, x='column_name', y='column_name')

import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')

sns.lineplot(data=data, x='sepal_length', y='sepal_width')
plt.title('Iris data analysis 1')
plt.show()
'''

#                 SET STYLE (changing axis style)
'''
set_style() method is used to set the aesthetic of the plot. 
It means it affects things like the color of the axes, whether 
the grid is active or not, or other aesthetic elements. 

There are five themes available in Seaborn.

darkgrid
whitegrid
dark
white
ticks

import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')

sns.set_style(style='dark')
sns.lineplot(data=data, x='sepal_length', y='sepal_width')

plt.suptitle('Whole Iris Data Analysis')
plt.show()
'''

#          CHANGING FIG SIZE IN SEABORN
'''

plt.figure(figsize=(10,3)) this should be placed before seaborn chart


import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')

plt.figure(figsize=(10,3))
sns.set_style(style='dark')
sns.lineplot(data=data, x='sepal_length', y='sepal_width')


plt.suptitle('Whole Iris Data Analysis')
plt.show()
'''

#           SET_CONTEXT(Customize size of labels, lines)
'''
It allows us to override default parameters. 
This affects things like the size of the labels, 
lines, and other elements of the plot, 
but not the overall style. The base context is “notebook”, 
and the other contexts are “paper”, “talk”, and “poster”. 
font_scale sets the font size.

EX:
import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')

sns.lineplot(data=data, x='sepal_length', y='sepal_width')
sns.set_context(context="poster", font_scale=1)
plt.suptitle('Whole Iris Data Analysis')
plt.show()
'''

#     SETTING TEMPORARY STYLE WITH WITH KEYWORD AND AXIS_STYLE
'''
import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')

def plot():
    sns.lineplot(data=data, x='sepal_length', y='sepal_width')

with sns.axes_style('darkgrid'):
     plt.subplot(211)
     plot()

plt.subplot(212)
plot()
plt.show()
'''

#               COLOR PALETTE
'''
Used to give different colors for each data points.

color_palette() method is used to give colors to the plot.
Another function palplot() is used to deal with the color palettes 
and plots the color palette as a horizontal array.

EX:
import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')
palette = sns.color_palette()

sns.palplot(palette)
plt.show()


TWO ARGUMENTS:
color_palette(name_of_predefined_color_palette, no_of_colour_want_from_predefined)

EX:
import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')
palette = sns.color_palette('muted',11)

sns.palplot(palette)
plt.show()


                    COLOR AS FIRST ARGUMENT

we can use give specific color in first argument

Sequential (light to dark or vice versa): 
'viridis', 'plasma', 'inferno', 'magma', 'cividis', 
'Greys', 'Blues', 'Reds', 'YlGnBu', Greens


EX:

import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')
palette = sns.color_palette('Greens',5)

sns.palplot(palette)
plt.show()


                  SETTING DFAULT COLOR PALATE


import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')

def plot():
    sns.lineplot(x="sepal_length", y="sepal_width", data=data)

sns.set_palette('vlag')
plt.subplot(211)
plot()

sns.set_palette('Accent')
plt.subplot(212)
plot()
plt.show()
'''

#                        BAR PLOT
'''
A barplot is basically used to aggregate the categorical data 
according to some methods and by default it is mean. 
It can also be understood as a visualization of the group by 
action. To use this plot we choose a categorical column for the x 
axis and a numerical column for the y axis and we see that it 
creates a plot taking a mean per categorical column. It can be 
created using the barplot() method.

import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')

sns.barplot(x="species",y='sepal_length',
            data=data)
plt.show()
'''

#                     COUNT PLOT
'''
A countplot basically counts the categories and returns a 
count of their occurrences. It is one of the most simple plots 
provided by the seaborn library. It can be created using the 
countplot() method.

import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')

sns.countplot(x="species",
            data=data)
plt.show()
'''

#                  BOX PLOT
'''
Same as matplot lib plot, gives distribution of quantitative data.

import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')

sns.boxplot(data=data,x="species", y='sepal_length',
             hue_order='species',color='gold') # hue_order sorts the data by the values of the column
plt.show()
'''

#                   VIOLIN PLOT
'''

A violin plot shows the distribution's shape (probability density) 
in addition to the box plot elements.

Example: A box plot might show the median income of 
different cities, while a violin plot would show the distribution 
of incomes within each city, revealing if it's normally distributed, 
skewed, or has multiple peaks.

EX:

import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')

sns.set_theme(context='paper', style='darkgrid')
sns.violinplot(data=data,x="species", y='sepal_width',
             hue_order='species', color='gold')
plt.show()
'''

#             STRIP PLOT
'''
It is looks like scatter plot

import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')

sns.set_theme(context='paper', style='darkgrid')
sns.stripplot(data=data,x="species", y='sepal_width',
             hue_order='species', color='gold')
plt.show()
'''

#               SWARM PLOT
'''

It will looks like drawing the scatter plot as violin plot, so
it scaled the density in x axis, it can be combined with violin plot

import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')

sns.set_theme(context='paper', style='darkgrid')
sns.swarmplot(data=data,x="species", y='sepal_width',
             hue_order='species', color='gold')
plt.show()
'''

#            HEATMAP
'''
Helps to understand the correlation between two variables, it
it mainly used for linear regression model.

PARAMETERS IT CAN TAKE:

Parameter | What It Does
data | The data to plot (like a correlation matrix: df.corr())
annot=True | Shows the actual numbers inside the cells
fmt='.2f' | Format of the numbers (e.g., 2 decimal places)
cmap='coolwarm' | Color scheme (many options: 'viridis', 'plasma', 'YlGnBu', etc.)
linewidths=0.5 | Adds space between cells
linecolor='black' | Color of the cell borders
cbar=True | Shows the color scale bar on the side (you can set False to hide it)
vmin & vmax | Set the min and max values for the color scale (helps compare multiple heatmaps)
xticklabels / yticklabels | Show/hide or customize row/column labels


import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('iris')

sns.heatmap(data[['sepal_length','sepal_width','petal_length']].corr(), 
            annot=True,fmt='.2f', linewidths=2,
            linecolor='black')
plt.show()
'''


#             SCATTER PLOT WITH HUE
'''
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=df, x='Temp_C', y='Rel Hum_%', hue='Months')
plt.title('Temperature vs Humidity (Colored by Month)')
plt.show()

'''


#                   FACET GRID
'''
Helps to visualize and combining categorical values with some column in 
2D space, that is called grid or facet grid.

import seaborn as sns
import matplotlib.pyplot as plt

# Load example dataset
tips = sns.load_dataset("tips")

# Create a FacetGrid to show histograms of total_bill by gender
g = sns.FacetGrid(tips, col="sex")
g.map(plt.hist, "total_bill", bins=10)

plt.show()

# Facet by both 'sex' and 'time'
g = sns.FacetGrid(tips, row="sex", col="time")
g.map(sns.scatterplot, "total_bill", "tip")

plt.show()

'''









