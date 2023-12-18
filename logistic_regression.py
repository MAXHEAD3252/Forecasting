# Logistic regression 
# Logistic regression aims to solve classification problems.
# It does this by predicting categorical outcomes, unlike linear regression that predicts a continuous outcome.

# Standard operational package imports.
import numpy as np
import pandas as pd

# Important imports for preprocessing, modeling, and evaluation.
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

# Visualization package imports.
import matplotlib.pyplot as plt
import seaborn as sns


df_original = pd.read_csv("F:\Learning_Work\Vs_Work\DM_Project\Invistico_Airline.csv")

# 10 rows shown
print(df_original.head(n=10))

print('\n ||||--------------------------------------------------------------------------|||| \n')
print('Type of data in the dataset...\n')

# data types of the data
print(df_original.dtypes)

# to check the number of satisfied customers in the dataset....
print('\n ||||--------------------------------------------------------------------------|||| \n')
print('Check the number of satisfied customers in the dataset...\n')
print(df_original['satisfaction'].value_counts(dropna = False))

# Check the number of satisfied customers in the dataset.....
print('\n ||||--------------------------------------------------------------------------|||| \n')
print('Check the number of satisfied customers in the dataset...\n')
print(df_original.isnull().sum())

# Drop the missing values
print('\n ||||--------------------------------------------------------------------------|||| \n')
print('Drop the rows with missing values...\n')
df_subset = df_original.dropna(axis=0).reset_index(drop = True)
print(df_subset)

# Prepare the data 
df_subset.astype({"Inflight entertainment": float})

#Convert the categorical column satisfaction into numeric
df_subset['satisfaction'] = OneHotEncoder(drop='first').fit_transform(df_subset[['satisfaction']]).toarray()

# data show after preperation
print('\n ||||--------------------------------------------------------------------------|||| \n')
print('Printing the whole dataset after preparation...\n')
print(df_subset.head(10))


# Create the training and testing data
# I put 70% of the data into a training set and the remaining 30% into a testing set.
X = df_subset[["Inflight entertainment"]]
y = df_subset["satisfaction"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
    
# Model building
#Fit a LogisticRegression model to the data
clf = LogisticRegression().fit(X_train,y_train)

# obtain parameters estimates
clf.coef_
clf.intercept_

# Create a plot of your model
# The graph seems to indicate that the higher the inflight entertainment value, the higher the customer satisfaction, 
sns.regplot(x="Inflight entertainment", y="satisfaction", data=df_subset, logistic=True, ci=None)
plt.show()

# Results and evaluation
# Save predictions.
print('\n ||||--------------------------------------------------------------------------|||| \n')
print('Printing the pridiction of test dataset...\n')
y_pred = clf.predict(X_test)
print(y_pred)

# Use the predict_proba and predict functions on X_test
# Use predict_proba to output a probability.
print('\n ||||--------------------------------------------------------------------------|||| \n')
print('Printing the probability of the test dataset...\n')
print(clf.predict_proba(X_test))

# Result Analysis 
print('\n ||||--------------------------------------------------------------------------|||| \n')
print('Result Analysis...\n')
print('Printing the overall accuracy of the algorithm...\n')
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, y_pred))
print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred))
print("Recall:", "%.6f" % metrics.recall_score(y_test, y_pred))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, y_pred))

print('\n ||||--------------------------------------------------------------------------|||| \n')
print('Logistic regression accurately predicted satisfaction ""80.2"" percent of the time.\n')

print('Customers who rated in-flight entertainment highly were more likely to be satisfied.Improving in-flight entertainment should lead to better customer satisfaction.\n')
print('The model is 80.2 percent accurate. This is an improvement over the datasets customer satisfaction rate.\n')