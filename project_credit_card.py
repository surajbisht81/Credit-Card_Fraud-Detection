
# - PROJECT ON CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING
# - DATASET TAKEN FROM KAGGLE
# - CREATED BY SURAJ BISHT


#-------------------------------------------------------------------------
# Step : 1
# Importing pandas and numpy library
# Read the data
#-------------------------------------------------------------------------
import pandas as pd
import numpy as np

#Reading the csv file
dataset = pd.read_csv('creditcard_2.csv')

# Checking for any null value in dataset
dataset.isnull().sum(axis=0)

# Viewing the head and tail part of the datset
print('Head part:\n')
dataset.head(5)
print('Tail part:\n')
dataset.tail(5)

# Splitting the Dataset into X(independent) and Y(dependent) Variables
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]


#-------------------------------------------------------------------------
# Step : 2
# visualising the data 
#-------------------------------------------------------------------------

# Counting the total no of Noramal and Fraud Cases
fraud = dataset[dataset['Class']==1]
normal = dataset[dataset['Class']==0]

# Another dataset containing only cases column
df = dataset['Class']


# Importing matplotlib library for ploting the data
# and Counter library for counting total no. of fraud and normal cases
import matplotlib.pyplot as plt
from collections import Counter
cases_count = Counter(df)

cases_value = list(cases_count.values())
fraud_class = ['Normal', 'Fraud']

# Plotting the pie chart
plt.pie(cases_value, labels=fraud_class, autopct='%.2f%%')
plt.show()

# Relation between Amount and Cases
plt.subplot(1,2,1)
plt.title('Amount vs Cases')
plt.scatter(dataset['Amount'], dataset['Class'], s=2)

# Relation between Amount and Time
plt.title('Amount vs Time')
plt.scatter(dataset['Amount'], dataset['Time'], s=2)

print(fraud.shape, normal.shape)

fraud.Amount.describe()
normal.Amount.describe()







#-----------------------------------------------------------------------
# Step : 3
# Model selection
#-----------------------------------------------------------------------

# Import and Train logistic regression classifier
from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression(random_state=1234)

# Import and Train Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=1234)

# Import and Train Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1234)

# Import and Train Support Vector Classifier
from sklearn.svm import SVC
svc = SVC(kernel='rbf', gamma=0.5)

# Perform Cross Validation
from sklearn.model_selection import cross_validate
cv_result_dtc = cross_validate(dtc, X, Y, cv=10, return_train_score=True)
cv_result_rfc = cross_validate(rfc, X, Y, cv=10, return_train_score=True)
cv_result_svc = cross_validate(svc, X, Y, cv=10, return_train_score=True)
cv_result_lrc = cross_validate(lrc, X, Y, cv=10, return_train_score=True)



# Get average of results

dtc_test_avg = np.average(cv_result_dtc['test_score'])
rfc_test_avg = np.average(cv_result_rfc['test_score'])
svc_test_avg = np.average(cv_result_svc['test_score'])
lrc_test_avg = np.average(cv_result_lrc['test_score'])


dtc_train_avg = np.average(cv_result_dtc['train_score'])
rfc_train_avg = np.average(cv_result_rfc['train_score'])
svc_train_avg = np.average(cv_result_svc['train_score'])
lrc_train_avg = np.average(cv_result_lrc['train_score'])


# printing the result as report
print()
print()
print('     ', 'Decision Tree ',  'Random Forest', 'Support Vector', 'Logistic Regression')
print('     ', '--------------', '------------', '--------------', '--------------------')

print('Test :',
      round(dtc_test_avg, 4), '      ',
      round(rfc_test_avg, 4), '      ',
      round(svc_test_avg, 4), '      ',
      round(lrc_test_avg,4))

print('Train :', 
      round(dtc_train_avg, 4), '        ',
      round(rfc_train_avg, 4), '         ',
      round(svc_train_avg, 4), '         ',
      round(lrc_train_avg, 4))





#--------------------------------------------------
# Import RandomizedSearchCV for Logistic Regression For Best Parameters

from sklearn.model_selection import RandomizedSearchCV

lrc_param = {'C':[0.01, 0.1, 0.5, 1, 5, 10],
             'penalty':['l2'],
             'solver':['liblinear', 'lbfgs', 'saga'],
             }

lrc_rs = RandomizedSearchCV(estimator=lrc,
                        param_distributions=lrc_param,
                        scoring='accuracy',
                        cv=10,
                        n_iter=10,
                        return_train_score=True,
                        random_state=1234)

lrc_rs_fit = lrc_rs.fit(X, Y)

param_result_lrc = pd.DataFrame.from_dict(lrc_rs_fit.cv_results_)


# best combination of parameters
print('\n the Best Parameters are :')
print(lrc_rs_fit.best_params_)



#-------------------------------------------------------------------------
# Step : 4
# Predicting the value and Analyzing the Model(Logistic Regression)
#-------------------------------------------------------------------------

# Step : 4.1
#-------------------------------------------------------------------------
# Splitting the dataset into  test and Train

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =      \
    train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)

lr = LogisticRegression(solver='liblinear', penalty='l2', C=0.01)

lr.fit(X_train, Y_train)

Y_predict = lr.predict(X_test)



# Step : 4.2
#-------------------------------------------------------------------------
# Build the Confusion Matrix and get the accuracy/Score

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(Y_test, Y_predict)

score = lr.score(X_test, Y_test)

cr = classification_report(Y_test, Y_predict)





















































































