import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from RandomForest import RandomForest

diab_df = pd.read_csv('diabetes.csv')


diab_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diab_df[[
    'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

diab_df['Glucose'].fillna(diab_df['Glucose'].mean(), inplace=True)
diab_df['BloodPressure'].fillna(diab_df['BloodPressure'].mean(), inplace=True)
diab_df['SkinThickness'].fillna(
    diab_df['SkinThickness'].median(), inplace=True)
diab_df['Insulin'].fillna(diab_df['Insulin'].median(), inplace=True)
diab_df['BMI'].fillna(diab_df['BMI'].median(), inplace=True)

x = diab_df.drop(['Outcome'], axis=1).values
y = diab_df['Outcome'].values


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


clf = RandomForest(X=X_train, y=y_train, n_trees=20)
# model = clf.fit(X_train, y_train)

with open('model', 'wb') as file:
    pickle.dump(clf, file)
