
import warnings 
warnings.filterwarnings('ignore')


import os
import numpy as np
import pandas as pd
import re
import string
from collections import Counter
import time


import missingno as msno
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
from pywaffle import Waffle






'''load the dataset'''
dat = pd.read_csv("healthcare-dataset-stroke-data.csv")




'''Plot the percentage graph'''
x = pd.DataFrame(dat.groupby(['stroke'])['stroke'].count())


#plot the subplot and add legends to the plot
fig, ax = plt.subplots(figsize = (6,6), dpi = 70)
ax.barh([1], x.stroke[1], height = 0.7, color = '#343bfe')
plt.text(-1150,-0.08, 'Healthy',{'font': 'Serif','weight':'bold','Size': '12','style':'normal', 'color':'#e6a129'})
plt.text(5000,-0.08, '95%',{'font':'Serif','weight':'bold' ,'size':'16','color':'#e6a129'})
ax.barh([0], x.stroke[0], height = 0.7, color = '#e6a129')
plt.text(-1000,1, 'Stroke', {'font': 'Serif','weight':'bold','Size': '12','style':'normal', 'color':'#343bfe'})
plt.text(300,1, '5%',{'font':'Serif', 'weight':'bold','size':'16','color':'#343bfe'})

#fill out the graphs with the following chosed color
fig.patch.set_facecolor('#f6f5f5')
ax.set_facecolor('#f6f5f5')


#Add legend to the bar plot
plt.text(-1150,1.77, 'Percentage of People Having Strokes and without strokes' ,{'font': 'Serif', 'Size': '18','weight':'bold', 'color':'black'})

plt.text(4650,0.8, 'Stroke ', {'font': 'Serif','weight':'bold','Size': '12','weight':'bold','style':'normal', 'color':'#343bfe'})

plt.text(5650,0.8, '|', {'color':'black' , 'size':'12', 'weight': 'bold'})

plt.text(5750,0.8, 'Healthy', {'font': 'Serif','weight':'bold', 'Size': '12','style':'normal', 'weight':'bold','color':'#e6a129'})

plt.text(-1150,1.5, 'We can see that it is a significantly unbalanced distribution,\nand clearly we see that 5 percent of people are likely to get \nheart strokes.', 
        {'font':'Serif', 'size':'12.5','color': 'black'})


#Use the plt function to set x-axis and y-axis and save the image. 
plt.tight_layout()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig("Percentage plot")