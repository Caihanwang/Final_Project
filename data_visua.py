
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





#Create numerical variables

dat['bmi_cat'] = pd.cut(dat['bmi'], bins = [0, 19, 25,30,10000], labels = ['Underweight', 'Ideal', 'Overweight', 'Obesity'])
dat['age_cat'] = pd.cut(dat['age'], bins = [0,13,18, 45,60,200], labels = ['Children', 'Teens', 'Adults','Mid Adults','Elderly'])
dat['glucose_cat'] = pd.cut(dat['avg_glucose_level'], bins = [0,90,160,230,500], labels = ['Low', 'Normal', 'High', 'Very High'])





'''plot Heart stroke and age graph'''
fig = plt.figure(figsize = (24,10), dpi = 60)

gt = fig.add_gridspec(10,24)
gt.update(wspace = 1, hspace = 0.05)


ax1 = fig.add_subplot(gt[1:10,13:]) 
ax2 = fig.add_subplot(gt[1:4,0:8]) 
ax3 = fig.add_subplot(gt[6:9, 0:8]) 


# set up axes list
axes = [ ax1,ax2, ax3]


# setting of axes
for ax in axes:
    ax.axes.get_yaxis().set_visible(False)
    ax.set_facecolor('#f6f5f5')
    
    for loc in ['left', 'right', 'top', 'bottom']:
        ax.spines[loc].set_visible(False)

fig.patch.set_facecolor('#f6f5f5')
        
ax1.axes.get_xaxis().set_visible(False)
ax1.axes.get_yaxis().set_visible(True)



stroke_age = dat[dat['stroke'] == 1].age_cat.value_counts()
healthy_age = dat[dat['stroke'] == 0].age_cat.value_counts()

ax1.hlines(y = ['Children', 'Teens', 'Adults', 'Mid Adults', 'Elderly'], xmin = [644,270,1691,1129,1127], 
          xmax = [1,1,11,59,177], color = 'grey',**{'linewidth':0.5})


sns.scatterplot(y = stroke_age.index, x = stroke_age.values, s = stroke_age.values*2, color = '#343bfe', ax= ax1, alpha = 1)
sns.scatterplot(y = healthy_age.index, x = healthy_age.values, s = healthy_age.values*2, color = '#e6a129', ax= ax1, alpha = 1)

ax1.axes.get_xaxis().set_visible(False)
ax1.set_xlim(xmin = -500, xmax = 2250)
ax1.set_ylim(ymin = -1,ymax = 5)

ax1.set_yticklabels( labels = ['Children', 'Teens', 'Adults', 'Mid Adults', 'Elderly'],fontdict = {'font':'Serif', 'fontsize':16,'fontweight':'bold', 'color':'black'})

ax1.text(-950,5.8, 'How Age Impact on Heart Strokes' ,{'font': 'Serif', 'Size': '25','weight':'bold', 'color':'black'},alpha = 0.9)
ax1.text(1000,4.8, 'Stroke ', {'font': 'Serif','weight':'bold','Size': '16','weight':'bold','style':'normal', 'color':'#343bfe'})
ax1.text(1300,4.8, '|', {'color':'black' , 'size':'16', 'weight': 'bold'})
ax1.text(1350,4.8, 'Healthy', {'font': 'Serif','weight':'bold', 'Size': '16','style':'normal', 'weight':'bold','color':'#e6a129'})
ax1.text(-950,5., 'Age have significant association with stokes, older people have larger probability of getting strokes \nmid age adults are the second higest', 
        {'font':'Serif', 'size':'16','color': 'black'})

ax1.text(stroke_age.values[0] + 30,4.05, stroke_age.values[0], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#343bfe'})
ax1.text(healthy_age.values[2] - 300,4.05, healthy_age.values[2], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#e6a129'})

ax1.text(stroke_age.values[1] + 30,3.05, stroke_age.values[1], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#343bfe'})
ax1.text(healthy_age.values[1] - 300,3.05, healthy_age.values[1], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#e6a129'})


# plot the distribution plots 

sns.kdeplot(data = dat, x = 'age', ax = ax2, shade = True, color = '#c76a48', alpha = 1, )
ax2.set_xlabel('Age of a person', fontdict = {'font':'Serif', 'color': 'black', 'size': 16,'weight':'bold' })
ax2.text(-17,0.025,'Age Distribution', {'font':'Serif', 'color': 'black','weight':'bold','size':24}, alpha = 0.9)
ax2.text(-17,0.021, 'From this graph we have adult population is the median group.', 
        {'font':'Serif', 'size':'16','color': 'black'})
ax2.text(80,0.019, 'Total',{'font':'Serif', 'size':'14','color': '#c76a48','weight':'bold'})
ax2.text(92,0.019, '=',{'font':'Serif', 'size':'14','color': 'black','weight':'bold'})
ax2.text(97,0.019, 'Stroke',{'font':'Serif', 'size':'14','color': '#343bfe','weight':'bold'})
ax2.text(113,0.019, '+',{'font':'Serif', 'size':'14','color': 'black','weight':'bold'})
ax2.text(117,0.019, 'Healthy',{'font':'Serif', 'size':'14','color': '#e6a129','weight':'bold'})


# plot the distribution plots and add legend and comments to the graph


sns.kdeplot(data = dat[dat['stroke'] == 0], x = 'age',ax = ax3, shade = True,  alpha = 1, color = '#e6a129' )
sns.kdeplot(data = dat[dat['stroke'] == 1], x = 'age',ax = ax3, shade = True,  alpha = 0.8, color = '#343bfe')

ax3.set_xlabel('Age of a person', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})

ax3.text(-17,0.0525,'Stroke-Age Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':24}, alpha= 0.9)
ax3.text(-17,0.043,'From the Distribution plot it is clear that old people are \nhaving larger number of strokes than young people.', {'font':'Serif', 'color': 'black', 'size':14})
ax3.text(100,0.043, 'Stroke ', {'font': 'Serif','weight':'bold','Size': '16','weight':'bold','style':'normal', 'color':'#343bfe'})
ax3.text(117,0.043, '|', {'color':'black' , 'size':'16', 'weight': 'bold'})
ax3.text(120,0.043, 'Healthy', {'font': 'Serif','weight':'bold', 'Size': '16','style':'normal', 'weight':'bold','color':'#e6a129'})

fig.text(0.25,0.05,'Relationship between Heart Strokes and Age',{'font':'Serif', 'weight':'bold','color': 'black', 'size':30})
plt.tight_layout()
plt.savefig("Heart stroke and age")




'''Plot the Heart stroke and glucose graph'''
fig = plt.figure(figsize = (24,10), dpi = 60)

gs = fig.add_gridspec(10,24)
gs.update(wspace = 1, hspace = 0.05)


ax2 = fig.add_subplot(gs[0:3,0:10]) 
ax3 = fig.add_subplot(gs[5:10, 0:10]) 
ax1 = fig.add_subplot(gs[0:,13:]) 

# setting up axes list
axes = [ ax1,ax2, ax3]


# setting of axes
for ax in axes:
    ax.axes.get_yaxis().set_visible(False)
    ax.set_facecolor('#f6f5f5')
    
    for loc in ['left', 'right', 'top', 'bottom']:
        ax.spines[loc].set_visible(False)

fig.patch.set_facecolor('#f6f5f5')
        
ax1.axes.get_xaxis().set_visible(False)
ax1.axes.get_yaxis().set_visible(True)


#plot of stoke and healthy people

stroke_glu = dat[dat['stroke'] == 1].glucose_cat.value_counts()
healthy_glu = dat[dat['stroke'] == 0].glucose_cat.value_counts()

ax1.hlines(y = ['Low', 'Normal', 'High', 'Very High'], xmin = [2316,1966,478,101], 
          xmax = [89,71,71,18], color = 'grey',**{'linewidth':0.5})


sns.scatterplot(y = stroke_glu.index, x = stroke_glu.values, s = stroke_glu.values, color = '#343bfe', ax= ax1, alpha = 1)
sns.scatterplot(y = healthy_glu.index, x = healthy_glu.values, s = healthy_glu.values, color = '#e6a129', ax= ax1, alpha = 1)

ax1.axes.get_xaxis().set_visible(False)
ax1.set_xlim(xmin = -500, xmax = 3000)
ax1.set_ylim(ymin = -1.5,ymax = 4.5)

ax1.set_yticklabels( labels = ['Low', 'Normal', 'High', 'Very High'],fontdict = {'font':'Serif', 'fontsize':16,'fontweight':'bold', 'color':'black'})

ax1.text(-1000,4.3, 'How Glucose level Impact on Heart Strokes' ,{'font': 'Serif', 'Size': '25','weight':'bold', 'color':'black'})
ax1.text(1700,3.5, 'Stroke ', {'font': 'Serif','weight':'bold','Size': '16','weight':'bold','style':'normal', 'color':'#343bfe'})
ax1.text(2050,3.5, '|', {'color':'black' , 'size':'16', 'weight': 'bold'})
ax1.text(2075,3.5, 'Healthy', {'font': 'Serif','weight':'bold', 'Size': '16','style':'normal', 'weight':'bold','color':'#e6a129'})
ax1.text(-1000,3.8, 'Glucose level does not have significant association with strokes.', 
        {'font':'Serif', 'size':'16','color': 'black'})


ax1.text(stroke_glu.values[0] + 30,0.05, stroke_glu.values[0], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#343bfe'})
ax1.text(healthy_glu.values[0] + -355,0.05, healthy_glu.values[0], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#e6a129'})

ax1.text(stroke_glu.values[2] + 30,1.05, stroke_glu.values[2], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#343bfe'})
ax1.text(healthy_glu.values[2] + 1170,1.05, healthy_glu.values[2], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#e6a129'})

ax1.text(stroke_glu.values[1] + 30,2.05, stroke_glu.values[1], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#343bfe'})
ax1.text(healthy_glu.values[1] - 1450,2.05, healthy_glu.values[1], {'font':'Serif', 'Size':14, 'weight':'bold', 'color':'#e6a129'})



# plotting distribution plots

sns.kdeplot(data = dat, x = 'avg_glucose_level', ax = ax2, shade = True, color = '#c76a48', alpha = 1, )
ax2.set_xlabel('Average Glucose Level', fontdict = {'font':'Serif', 'color': 'black', 'size': 16,'weight':'bold' })
ax2.text(25,0.023,'Glucose Distribution', {'font':'Serif', 'color': 'black','weight':'bold','size':20})
ax2.text(25,0.019, 'From the distribution plot, we see most people have similar glocose level.', 
        {'font':'Serif', 'size':'16','color': 'black'})
ax2.text(210,0.017, 'Total',{'font':'Serif', 'size':'14','color': '#c76a48','weight':'bold'})
ax2.text(240,0.017, '=',{'font':'Serif', 'size':'14','color': 'black','weight':'bold'})
ax2.text(250,0.017, 'Stroke',{'font':'Serif', 'size':'14','color': '#343bfe','weight':'bold'})
ax2.text(280,0.017, '+',{'font':'Serif', 'size':'14','color': 'black','weight':'bold'})
ax2.text(290,0.017, 'Healthy',{'font':'Serif', 'size':'14','color': '#e6a129','weight':'bold'})


# distribution plots adding comments and legends


sns.kdeplot(data = dat[dat['stroke'] == 0], x = 'avg_glucose_level',ax = ax3, shade = True,  alpha = 1, color = '#e6a129' )
sns.kdeplot(data = dat[dat['stroke'] == 1], x = 'avg_glucose_level',ax = ax3, shade = True,  alpha = 0.8, color = '#343bfe')

ax3.set_xlabel('Average Glucose Level', fontdict = {'font':'Serif', 'color': 'black', 'weight':'bold','size': 16})

ax3.text(-17,0.0195,'Stroke-Glucose Distribution', {'font':'Serif', 'weight':'bold','color': 'black', 'size':20})
ax3.text(-17,0.0176,'It is hard to determine whether glucose level effect \npeople of having strokes.', {'font':'Serif', 'color': 'black', 'size':14})
ax3.text(240,0.0174, 'Stroke ', {'font': 'Serif','weight':'bold','Size': '16','weight':'bold','style':'normal', 'color':'#343bfe'})
ax3.text(290,0.0174, '|', {'color':'black' , 'size':'16', 'weight': 'bold'})
ax3.text(300,0.0174, 'Healthy', {'font': 'Serif','weight':'bold', 'Size': '16','style':'normal', 'weight':'bold','color':'#e6a129'})


fig.text(0.2,0.03,'Assocaition between Heart Strokes and Glucose',{'font':'Serif', 'weight':'bold','color': 'black', 'size':25})
plt.tight_layout()
plt.savefig("Heart stroke and glutose")

