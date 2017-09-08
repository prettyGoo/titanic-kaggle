import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn import preprocessing

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from simplifiers import *
from visio import *


def encode_features(df_train, df_test):
    features = ['Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix', 'Embarked', 'Parch', 'Fare']
    df_combined = pd.concat([df_train[features], df_test[features]])

    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test


if __name__ == '__main__':
    data_train = pd.read_csv('train.csv')
    data_test = pd.read_csv('test.csv')

    data_train = transform_features(data_train)
    data_test = transform_features(data_test)

    # visio(data_train)

    data_train, data_test = encode_features(data_train, data_test)
    X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
    y_all = data_train['Survived']

    num_test = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test)

    # clf = RandomForestClassifier()
    # clf = AdaBoostClassifier()
    # clf = GradientBoostingClassifier()

    # clf = DecisionTreeClassifier()
    # clf = BaggingClassifier(KNeighborsClassifier())
    # clf = KNeighborsClassifier()
    br = BernoulliNB()
    rf = RandomForestClassifier()
    ada = AdaBoostClassifier()
    dt = DecisionTreeClassifier()
    kn = KNeighborsClassifier()
    gb = GradientBoostingClassifier()

    bg1 = BaggingClassifier(br)
    bg2 = BaggingClassifier(ada)
    bg3 = BaggingClassifier(rf)
    bg4 = BaggingClassifier(dt)
    bg5 = BaggingClassifier(gb)
    # clf = VotingClassifier(estimators=[('bg1', bg5), ('bg2', bg1), ('bg3', bg2)])

    # clf = VotingClassifier(estimators=[('br', BernoulliNB()), ('rf', RandomForestClassifier()), ('grd', AdaBoostClassifier())])

    # classifier 1
    clf = GradientBoostingClassifier()

    parameters = {
        'loss': ['deviance'],
        'learning_rate': [0.05, 0.06, 0.07, 0.085, 0.1],
        'n_estimators': [50, 70, 100],
        'max_depth': [1, 2, 3, 5]
        # 'voting': ['soft', 'hard']
        # 'alpha': [0.1, 0.2, 0.5, 0.65, 0.7, 0.75, 0.85, 1, 2, 3, 10],
        # 'n_estimators': [6, 7, 8, 9, 10, 11, 20, 100],
        # 'algorithm': ['SAMME', 'SAMME.R']
        # 'n_neighbors': [1, 2, 5, 7, 10]

        # 'max_features': ['log2', 'sqrt', 'auto'],
        # 'criterion': ['entropy', 'gini'],

        # 'max_samples': [10, 20, 50, 100],
        # 'max_features': [0.5, 0.7, 0.8, 0.85, 0.9, 1]
        # 'max_depth': [5, 6.75, 7, 9, 10],
        # 'presort': [True, False],
        # 'min_samples_split': [2, 3, 5],
        # 'min_samples_leaf': [1, 5, 8, 10]

        # 'learning_rate': [0.001, 0.02, 0.5, 0.1, 0.4, 0.8, 1],
        # 'min_samples_split': [0.002, 0.02, 0.1, 0.3, 0.5, 0.7, 0.91],
        # 'min_impurity_split': [1, 2, 5, 7, 10]
    }

    accuracy_scorer = make_scorer(accuracy_score)
    gs = GridSearchCV(clf, parameters, scoring=accuracy_scorer)
    gs = gs.fit(X_train, y_train)
    try:
        print(gs.best_estimator_)
    except Exception as e:
        print(e)

    clf1 = gs.best_estimator_

    # classifier 2
    clf = BernoulliNB()

    parameters = {
        'alpha': [0.1, 0.2, 0.5, 0.65, 0.7, 1, 3, 10],
        # 'n_estimators': [6, 7, 8, 9, 10, 11, 20, 100],
        # 'algorithm': ['SAMME', 'SAMME.R']
    }

    accuracy_scorer = make_scorer(accuracy_score)
    gs = GridSearchCV(clf, parameters, scoring=accuracy_scorer)
    gs = gs.fit(X_train, y_train)
    try:
        print(gs.best_estimator_)
    except Exception as e:
        print(e)

    clf2 = gs.best_estimator_

    # classifier 3
    clf = AdaBoostClassifier()

    parameters = {
        'base_estimator': [dt, rf],
        'n_estimators': [20, 50, 70, 100],
    }

    accuracy_scorer = make_scorer(accuracy_score)
    gs = GridSearchCV(clf, parameters, scoring=accuracy_scorer)
    gs = gs.fit(X_train, y_train)
    try:
        print(gs.best_estimator_)
    except Exception as e:
        print(e)

    clf3 = gs.best_estimator_

    clf = VotingClassifier(estimators=[('clf1', clf1), ('clf2', clf2), ('clf3', clf3)])
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    print(accuracy_score(y_test, predictions))

    # kf = KFold(shuffle=True, n_splits=10)
    # outcomes = []
    # iter = 0
    #
    # for train_index, test_index in kf.split(X_all, y_all):
    #     iter += 1
    #     X_train, X_test = X_all.values[train_index], X_all.values[test_index]
    #     y_train, y_test = X_all.values[train_index], X_all.values[test_index]
    #
    #     clf.fit(X_train, y_train)
    #     predictions = clf.predict(X_test)
    #     outcome = accuracy_scorer(y_test, predictions)
    #     outcomes.append(outcome)
    #
    # mean_outcome = np.mean(outcomes)
    # print(mean_outcome)

    # predictions = clf.predict(X_test)
    # print(accuracy_score(y_test, predictions))

    # POST TO THE KAGGLE
    ids = data_test['PassengerId']
    data = data_test.drop('PassengerId', axis=1)
    predictions = clf.predict(data)
    output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
    output.to_csv('titanic_predictions.csv', index=False)

