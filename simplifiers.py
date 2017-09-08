import pandas as pd


def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 4, 12, 18, 25, 35, 60, 110)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df


def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df


def simplify_parches(df):
    df.Parch = df.Parch.fillna(0)
    bins = (-1000, 0, 1, 2, 4, 6, 100)
    group_names = ['No Info', 'Single', 'Couple', 'Small Group', 'Group', 'Company']
    categories = pd.cut(df.Parch, bins, labels=group_names)
    df.Parch = categories
    return df


def simplify_embarkations(df):
    df.Embarked = df.Embarked.fillna('N')
    return df


def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df


def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df


def drop_features(df):
    return df.drop(['Ticket', 'Name'], axis=1)


def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = simplify_parches(df)
    df = simplify_embarkations(df)
    df = format_name(df)
    df = drop_features(df)
    return df