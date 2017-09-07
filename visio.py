import seaborn as sns
import matplotlib.pyplot as plt


def visio(df):
    sns.set_style('white')
    sns.despine(left=False)

    embarked(df)


def embarked(df):
    # sns.barplot(x='Embarked', y='Survived', hue='Sex', data=df.loc[df['Pclass'] == 1])
    # plt.show()
    plt.title('First-class survivors')

    # sns.barplot(x=df.loc[df['Pclass'] == 2, 'Embarked'], hue=df.loc[df.Pclass == 2, 'Sex'], y=df.loc[df['Pclass'] == 2, 'Survived'])
    # plt.show()
    plt.title('Second-class survivors')

    sns.barplot(x=df.loc[df['Pclass'] == 3, 'Embarked'], hue=df.loc[df.Pclass == 3, 'Sex'], y=df.loc[df['Pclass'] == 3, 'Survived'], palette={'male': 'blue', 'female': 'green'}, markers=['*', 'o'], linestyles=['-', '--'])
    plt.show('Third-class survivors')
