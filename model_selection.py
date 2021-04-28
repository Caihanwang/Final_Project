from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


'''input data'''
raw_data = pd.read_csv("healthcare-dataset-stroke-data.csv")




'''data cleaning'''
# drop ID variable
ds = raw_data.drop("id",axis=1)
# fill in the NA by mean BMI
ds["bmi"].fillna(ds["bmi"].mean(),inplace=True)



'''data preprocessing'''
ds_X = ds.drop("stroke",axis=1)
ds_y = ds.iloc[:,-1]
ds_Predict1 = ds_X

#encode data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
cat_encode = ColumnTransformer([('encoder', OneHotEncoder(), [0,5,9])], remainder= 'passthrough')
ds_X = cat_encode.fit_transform(ds_X)
ds_X = pd.DataFrame(ds_X)


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
ds_X[15] = encoder.fit_transform(ds_X[15])
ds_X[16] = encoder.fit_transform(ds_X[16])


#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(ds_X, ds_y, test_size= 0.2, random_state=42)

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#use SMOTE to enlarge sample size
from imblearn.over_sampling import SMOTE
samp = SMOTE(random_state=3)
X_train, y_train = samp.fit_resample(X_train, y_train.ravel())




'''model selection'''
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, average_precision_score, precision_recall_curve, plot_precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

#model 1. logistic regression model 
model_1 = LogisticRegression(random_state=42)
model_1.fit(X_train,y_train)
y_pred = model_1.predict(X_test)
print('Accuracy Score of logistic regression is', model_1.score(X_test, y_test))
print(classification_report(y_test, y_pred))

#model 2. Support Vector Machine
from sklearn.svm import SVC
model_2 = SVC(random_state=42)
model_2.fit(X_train,y_train)
y_pred = model_2.predict(X_test)
print('Accuracy Score of SVM is', model_2.score(X_test, y_test))
print(classification_report(y_test, y_pred))

#model 3. random forest
from sklearn.ensemble import RandomForestClassifier
model_3 = RandomForestClassifier(random_state=42)
model_3.fit(X_train,y_train)
y_pred = model_3.predict(X_test)
print('Accuracy Score of Random Forest is', model_3.score(X_test, y_test))

print(classification_report(y_test, y_pred))

#Random Forest have the highest accuracy score, so we choose to use Random Forest model


'''Predict Function'''

def Predict(ThePatient):
    # ThePatient = [gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smorking_status]

    #encode 
    ds_Predict = ds_Predict1
    ds_Predict.loc['ThePatient'] = {'gender':ThePatient[0], 'age':ThePatient[1], 'hypertension':ThePatient[2], 'heart_disease':ThePatient[3], 'ever_married':ThePatient[4], 'work_type':ThePatient[5], 'Residence_type':ThePatient[6], 'avg_glucose_level':ThePatient[7], 'bmi':ThePatient[8], 'smoking_status':ThePatient[9]}
    cat_encode = ColumnTransformer([('encoder', OneHotEncoder(), [0,5,9])], remainder= 'passthrough')
    ds_Predict = cat_encode.fit_transform(ds_Predict)
    ds_Predict = pd.DataFrame(ds_Predict)   
    encoder = LabelEncoder()
    ds_Predict[15] = encoder.fit_transform(ds_Predict[15])
    ds_Predict[16] = encoder.fit_transform(ds_Predict[16])
    ds_Predict = sc.transform(ds_Predict.astype(str))
   
    #preidct
    y = model_3.predict(ds_Predict[-1:])
    if y == [1]:
        return "The patient is stroke"
    if y == [0]:
        return "The patient is not stroke"


'''Test'''
Patient_A = ['Male', 67.0, 0, 1, 'Yes', 'Private', 'Urban', 228.69, 36.600000, 'formerly smoked']# expect: stroke
print(Predict(Patient_A))

Patient_B = ['Female', 44.0, 0, 0, 'Yes', 'Govt_job', 'Urban', 85.28, 26.2, 'Unknown']# expect: not stroke
print(Predict(Patient_B))

Patient_C = ['Male', 22.0, 0, 0, 'No', 'Private', 'Urban', 200, 22.85, 'never smoked']# expect: stroke
print(Predict(Patient_C))
# print(Predict(['Male', 78.0, 0, 1, 'No', 'Govt_job', 'Urban', 167.0, 45.0, 'smokes']))

