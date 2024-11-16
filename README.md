# Student Enrollment and Support Prediction
## The dataset contains the following columns:
Student ID
Name
Age
Grade Level
Program Interest
Enrollment Likelihood
Needs Support
Support Type


## Data Preprocessing:
Converts categorical data (Program Interest, Grade Level, Support Type) into numeric format using LabelEncoder.
Maps the Enrollment Likelihood column to binary values


### Model Training:
Uses a Random Forest Classifier to predict:
Whether a student needs support
The likelihood of enrollment
Splits the dataset into training and testing sets.


### Model Evaluation:
Displays classification reports for both models


### Filtered Data:
Displays students who are likely to enroll and need support.

Run the script, and it will output the evaluation of both models and the filtered students who are likely to enroll and need support.
