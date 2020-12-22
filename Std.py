# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier #Import Decision Tree Classifier
from sklearn.model_selection import train_test_split #Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#Load dataset
data = pd.read_csv("StudentsPerformance.csv")

data["average score"]=np.mean(data[['math score', 'reading score','writing score']],axis=1).round(1)
data['pass/fail']=np.where(data['average score']>50,1,0)

#split dataset in features and target variable
#convert_categorial = ['gender','race/ethnicity','parent level of education','lunch','test preparation course']
convert_cat = pd.get_dummies(data,prefix_sep="_", drop_first=True)
data = convert_cat.copy()
data.columns=['math_score','reading_score','writing_score','average_score','pass/fail','gender_male','race/ethnicity_group_B','race/ethnicity_group_C','race/ethnicity_group_D','race/ethnicity_group_E','parental_level_of_education_bachelor_degree',
              'parental_level_of_education_high_school','parental_level_of_education_master_degree','parental_level_of_education_some_college','parental_level_of_education_some_high_school','lunch_standard','test_preparation_course_none']
data.head()
feature_cols =['math_score','reading_score','writing_score','gender_male','race/ethnicity_group_B','race/ethnicity_group_C','race/ethnicity_group_D','race/ethnicity_group_E','parental_level_of_education_bachelor_degree',
              'parental_level_of_education_high_school','parental_level_of_education_master_degree','parental_level_of_education_some_college','parental_level_of_education_some_high_school','lunch_standard','test_preparation_course_none']
target = data['pass/fail']

#Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(data[feature_cols], target, test_size=0.2)
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(x_train,y_train)
cn = ['pass','fail']
#Predict labels of unseen (test) data
y_pred = clf.predict(x_test)
fig, axes = plt.subplots(figsize=(4,4),dpi=1600)
print("Accuracy : ",metrics.accuracy_score(y_test,y_pred))
tree.plot_tree(clf,feature_names = feature_cols,class_names=cn,filled=True);
fig.savefig('StudentsPerformance.png')
print(data['pass/fail'].value_counts())