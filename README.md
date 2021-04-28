# Stroke Prediction
Team member: Caihan Wang, Yifeng Tang, Jiajie Shen. 

## Description
We are going to create an application which could predict the stroke of patients, giving their Gender, Age, Hypertension, Heart Disease, Ever Married, Work Type, Residence Type, Avg. Glucose Level, BMI, Smoking Status.

## Plan
For our data visualization part, we will include multiple informative graphs to illustrate the valuable information from our dataset: Percentage of people having strokes(in bar plot showing the percentage of each category), Heart stroke and age, Heart stroke and Glucose, Heart stroke and weight, Gender risk for stroke, Hypertension Risk for stroke, Heart disease for stroke. These graphs can help collaborators to understand the association between stroke and different variables. It will be very helpful for the latter project to do model prediction to predict whether a person is going to have stroke or not. 

For our model part, We are going to choose a model from logistic regression, Random Forest, Decision tree. The model packages could be imported from sklearn. We clean and split data with test data and train data. After training model, We do the model diagnose by test data with statistical methods to find the most accurate one to adjust for our data. We save the best model and create a function with this model to predict the stroke status of patients. The input value of this function should be Gender, Age, Hypertension, Heart Disease, Ever Married, Work Type, Residence Type, Avg Glucose Level, BMI, Smoking Status. The output will be whether the patient has high probability to have stroke. Then we test the availabilty of the function. If there is no problem, we can wrap up the function with a interface.  

For interface, 



## Architecture
We used "healthcare-dataset-stroke-data.csv", which cited from Kaggle. We used the data to train sklearning models and predict the stroke status of patients. The whole process included data visualization, model selection, and user interface.  

### Data Visulization

<br>  



### Model Selection and Function
Firstly, we read and cleaned the data by removing all NA in BMI. In order to fit the model, we converted all the strings like ['Yes', 'No'] to [1, 0] and created dummy variables to form a new table.  
Secondly, for data preprocessiong, we found that the cleaned data are extremely unbalanced, because 4700 people were not stroke but 209 people were stroke in the data. If we use this data to fit the models, models will be extremely skewed to the 'Not Stroke'. Therefore, to get an accurate prediction model, we chose to decrease the sample size to a balanced one by randomly select 209 people from people who are not stroke. After that, our data was formed by 209 people who were stroke and 209 people who were not stroke. Then, we splitted the data to test data and train data.  
Thirdly, we imported model functions from sklearn, and chose to fit 3 models, including logistic regression model, random forest and decision tree. To diagnose models, we also created a funtion named 'generate_model_report'. After fitting, we could see that for logistic regression, the accuracy was 0.80 and the F1 score was 0.80. For random forest, the accuracy was 0.76 and the F1 score was 0.76. For decision treee, the accuracy was 0.71 and the F1 score was 0.72. Since logistic regression had the highest accuracy score and F1 score, so we chose to use Logistic Regression model and saved it as 'Model1.pkl'.  
At last, we created a function named 'Predict'. The input value should be a list including the patient's gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi and smorking_status. We imported 'Model1.pkl' in the function to predict the stroke status.  
<br>


### User Interface




## Instruction
To run the application, you should put the "healthcare-dataset-stroke-data.csv", "Model1.pkl", "Stroke_Prrediction.py", "test_Stroke_Prediction.py", "ModelSelection.py", "created.png", "Duke_Chapel.png", "duke.png", "data_visualization.py" into one folder. Also, you need install all the packages in "requirement.txt"  
Run "Stroke_Prediction.py", the user interface will show. You can choose and fill in some data in the interface, it will output the predict result.  

## Examples
![截屏2021-04-28 下午10.23.38.png](https://i.loli.net/2021/04/28/VmA3QkDqBar6hj7.png)
![截屏2021-04-28 下午10.24.13.png](https://i.loli.net/2021/04/28/6Ocjzy5EaQCLZNH.png)
![截屏2021-04-28 下午10.24.36.png](https://i.loli.net/2021/04/28/UP1bSQv3uzIlca8.png)
![截屏2021-04-28 下午10.24.46.png](https://i.loli.net/2021/04/28/S1YEsC6Ud4ciyMq.png)

## Limitation & Discussion
The number of people who are stroke is small, so that the model is not so accurate. And the predict result is only for reference and not absolute. If we can get the more comprehensive data, the model will be much better.  
