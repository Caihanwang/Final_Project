import pandas as pd
import joblib
Model2 = joblib.load('Model2.pkl')

def Predict(ThePatient):
    # ThePatient = [gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smorking_status]
    
    # create an empty dataframe
    ds = pd.DataFrame(columns=['Age', 'Male', 'HT_Yes', 'HD_Yes', 'EM_Yes', 'Never_worked', 'Private', 'Self-employed', 'children', 'BMI', 'Avg. Glucose Level',"RT_Urban","SS_formerly smoked","SS_never smoked","SS_smokes"])
    
    # fill in the dataframe by ThePatient
    ds['Age'] = [ThePatient[1]]

    if ThePatient[0] == 'Male':
        ds['Male'] = 1
    else:
        ds['Male'] = 0

    ds['HT_Yes'] = [ThePatient[2]]

    ds['HD_Yes'] = [ThePatient[3]]

    ds['EM_Yes'] = [ThePatient[4]]
    ds["EM_Yes"].replace(["No","Yes"], [0,1], inplace=True)

    if ThePatient[5] == 'Never_worked':
        ds['Never_worked'] = 1
    else:
        ds['Never_worked'] = 0
    if ThePatient[5] == 'Private':
        ds['Private'] = 1
    else:
        ds['Private'] = 0
    if ThePatient[5] == 'Self-employed':
        ds['Self-employed'] = 1
    else:
        ds['Self-employed'] = 0
    if ThePatient[5] == 'children':
        ds['children'] = 1
    else:
        ds['children'] = 0

    ds['BMI'] = [ThePatient[8]]

    ds['Avg. Glucose Level'] = [ThePatient[7]]
    
    if ThePatient[6] == 'Urban':
        ds['RT_Urban'] = 1
    else:
        ds['RT_Urban'] = 0

    if ThePatient[9] == 'formerly smoked':
        ds['SS_formerly smoked'] = 1
    else:
        ds['SS_formerly smoked'] = 0
    if ThePatient[9] == 'never smoked':
        ds['SS_never smoked'] = 1
    else:
        ds['SS_never smoked'] = 0
    if ThePatient[9] == 'smokes':
        ds['SS_smokes'] = 1
    else:
        ds['SS_smokes'] = 0


    #Model predict
    y = Model2.predict(ds)

    #Output
    if y == [1]:
        return "It is highly possible that the patient is stroke"
    if y == [0]:
        return "It is highly possible that the patient is not stroke"

