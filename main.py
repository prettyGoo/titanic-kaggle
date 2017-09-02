import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from simplifiers import *


if __name__ == '__main__':
    data_train = pd.read_csv('train.csv')

    # sns.barplot(x='Embarked', y='Survived', hue='Sex', data=data_train)
    # plt.show()
    #
    # sns.pointplot(x='Pclass', y='Survived', hue='Sex', data=data_train, palette={'male': 'blue', 'female': 'yellow'}, markers=['*', 'o'], linestyles=['-', '--'])
    # plt.show()

    data_train = simplify_ages(data_train)
    data_train = simplify_cabins(data_train)
    data_train = simplify_sex(data_train)
    print(data_train.sample(1))

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot(xs=data_train.Sex, ys=data_train.Survived, zs=data_train.Pclass)
    # plt.show()


    # sns.pointplot(ci=None, x='Age', y='Survived', hue='Sex', data=data_train, palette={'male': 'blue', 'female': 'green'}, markers=['*', 'o'], linestyles=['-', '--'])
    # plt.show()

    # sns.pointplot(ci=None, x='Cabin', y='Survived', hue='Sex', data=data_train)
    # plt.show()
    # X_all = data_train.drop(['Survived', 'PassengerId', 'Cabin', 'Embarked'], axis=1)
    X_all = data_train[['Sex', 'Age']]
    y_all = data_train.Survived

    num_test = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=20)

    clf = RandomForestClassifier()
    parameters = {
        'n_estimators': [4, 6, 9, 11],
        'max_features': ['log2', 'sqrt', 'auto'],
        'max_depth': [2, 3, 5, 7, 10],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 5, 8]
    }


    accuracy_scorer = make_scorer(accuracy_score)
    grid_obj = GridSearchCV(clf, parameters, scoring=accuracy_scorer)
    grid_obj = grid_obj.fit(X_train, y_train)


    clf = grid_obj.best_estimator_
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    print(accuracy_score(y_test, predictions))

