# Stroke Prediction
Team member: Caihan Wang, Yifeng Tang, Jiajie Shen. 
## Description
We are going to create an application which could predict the stroke of patients, giving their gender, age, hypertension and so on. 
## Architecture
We are going to use SQLites database to store the data, which cited from Kaggle, and use the data to train sklearning models and predict the stroke status of patients. The whole process includes data visualization, model selection, and user interface. We still haven't decided which method to build user interface.
## Plan
For our data visualization part, we will include multiple informative graphs to illustrate the valuable information from our dataset: Percentage of people having strokes(in bar plot showing the percentage), Heart stroke and age, Heart stroke and Glucose, Heart stroke and weight, Gender risk for stroke, Hypertension Risk for stroke, Heart disease for stroke, Marriage and stroke, Lifestyle and stroke, Smoking and stroke, Correlation Map of features - How closely each of the features correlated. These graphs can help collaborators to understand the association between stroke and different variables.  

For our model part, We are going to choose a model from logistic regression, SVM, Decision tree, Random forest and XGBoost. We do the model diagnose with statistical methods to find the most accurate one to adjust for our data. The model packages could be imported from sklearn.
