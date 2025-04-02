#    MARPLOTLIN SUBPLOTS

'''
 In Matplotlib, subplots allow you to create multiple plots 
 (charts, graphs, or visualizations) within a single figure. 
 This is especially useful when you want to compare multiple datasets 
 or present related visualizations side-by-side.

 Key Concepts

 Figure: The overall container for all plots.
 Axes: The individual plots within the figure.
 Using subplots, you can arrange multiple axes in a grid-like 
 layout within a figure


 plt.subplot()
 This creates a single subplot within a grid.
 Syntax: plt.subplot(nrows, ncols, index)
 nrows: Number of rows in the grid.
 ncols: Number of columns in the grid.
 index: The position of the current subplot (1-based indexing).

 import matplotlib.pyplot as plt

 # Create a figure with 2 rows and 1 column of subplots
 plt.subplot(2, 1, 1)  # First subplot
 plt.plot([1, 2, 3], [4, 5, 6])
 plt.title("First Plot")

 plt.subplot(2, 1, 2)  # Second subplot
 plt.bar([1, 2, 3], [3, 2, 1])
 plt.title("Second Plot")

 plt.tight_layout()  # Adjust spacing
 plt.show()

'''


#       MATPLOT LIB SUBPLOTS FIGSIZE

'''
 The figsize parameter in Matplotlib's plt.subplots() 
 (and other related functions like plt.figure()) is used 
 to set the size of the figure, which is the entire drawing 
 area containing all the subplots, axes, and decorations 
 (titles, labels, etc.).

 Syntax of figsize
 fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))

 width: The width of the figure in inches.
 height: The height of the figure in inches.

 Why Use figsize?
 
 To ensure the plots are properly sized and not too cramped.
 To improve readability of complex visualizations with many subplots.
 To adjust the figure dimensions for display or saving purposes.

 EX:

 import matplotlib.pyplot as plt

 # Create a figure with a specific size
 fig, axes = plt.subplots(2, 2, figsize=(10, 6))  # Width: 10 inches, Height: 6 inches

 # Add some plots
 axes[0, 0].plot([1, 2, 3], [4, 5, 6])
 axes[0, 0].set_title("Line Plot")

 axes[0, 1].bar([1, 2, 3], [3, 2, 1])
 axes[0, 1].set_title("Bar Chart")

 axes[1, 0].scatter([1, 2, 3], [3, 2, 1])
 axes[1, 0].set_title("Scatter Plot")

 axes[1, 1].hist([1, 1, 2, 3, 3, 3], bins=3)
 axes[1, 1].set_title("Histogram")

 # Adjust spacing
 plt.tight_layout()

 # Display the plots
 plt.show()

'''


#   CREATING BAR CHART WITH SERIES, SUNPLOTS FIGSIE

'''
 # Create a bar plot of class distributions
 fig, ax = plt.subplots(figsize=(10, 5))

 # Plot the data
 ax.bar(class_distributions.index, class_distributions.values)  # Write your code here
 ax.set_xlabel("Class Label")
 ax.set_ylabel("Frequency [count]")
 ax.set_title("Class Distribution, Multiclass Training Set")
 plt.xticks(rotation=45)
 plt.tight_layout()
 plt.show()

'''

#              CREATING NORMAL PLOT
'''
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.plot(10, 20)
plt.show()
'''
