import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

joseh = pd.read_csv('C:\\Users\\HomePC\\Desktop\\cat 2\\student_records.csv')
le_program = LabelEncoder()
joseh['Program Interest'] = le_program.fit_transform(joseh['Program Interest'])
le_grade = LabelEncoder()
joseh['Grade Level'] = le_grade.fit_transform(joseh['Grade Level'])
le_support = LabelEncoder()
joseh['Support Type'] = joseh['Support Type'].fillna('None')
joseh['Support Type'] = le_support.fit_transform(joseh['Support Type'])
joseh['Enrollment Likelihood'] = joseh['Enrollment Likelihood'].map({
    'High': 1,
    'Medium': 0,
    'Low': 0
})
X = joseh[['Age', 'Program Interest', 'Grade Level', 'Support Type']]
y_support = joseh['Needs Support']
X_train, X_test, y_train_support, y_test_support = train_test_split(X, y_support, test_size=0.3, random_state=42)
rf_support = RandomForestClassifier(n_estimators=100, random_state=42)
rf_support.fit(X_train, y_train_support)
y_pred_support = rf_support.predict(X_test)
print("Support Model Evaluation:")
print(classification_report(y_test_support, y_pred_support))
y_enrollment = joseh['Enrollment Likelihood']
X_train, X_test, y_train_enrollment, y_test_enrollment = train_test_split(X, y_enrollment, test_size=0.3, random_state=42)
rf_enrollment = RandomForestClassifier(n_estimators=100, random_state=42)
rf_enrollment.fit(X_train, y_train_enrollment)
y_pred_enrollment = rf_enrollment.predict(X_test)
print("\nEnrollment Model Evaluation:")
print(classification_report(y_test_enrollment, y_pred_enrollment))
filtered_data = joseh[(joseh['Enrollment Likelihood'] == 1) & (joseh['Needs Support'] == True)]
print("\nStudents Likely to Enroll and Need Support:")
print(filtered_data[['Student ID', 'Name', 'Age', 'Grade Level', 'Program Interest', 'Needs Support', 'Support Type']])
