import numpy as np # linear algebra
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt



def readData():
    d = pd.read_csv('../Code/dataFile/dataset.csv')
    return d


def analyseData(attrition):

    f, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=False, sharey=False)


    s = np.linspace(0, 3, 10)

    cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)
    x = attrition['Age'].values
    y = attrition['TotalWorkingYears'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=axes[0, 0])
    axes[0, 0].set(title='Age against Total working years')

    cmap = sns.cubehelix_palette(start=0.333333333333, light=1, as_cmap=True)
    x = attrition['Age'].values
    y = attrition['DailyRate'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0, 1])
    axes[0, 1].set(title='Age against Daily Rate')

    cmap = sns.cubehelix_palette(start=0.666666666667, light=1, as_cmap=True)
    x = attrition['YearsInCurrentRole'].values
    y = attrition['Age'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0, 2])
    axes[0, 2].set(title='Years in role against Age')

    cmap = sns.cubehelix_palette(start=1.0, light=1, as_cmap=True)
    x = attrition['DailyRate'].values
    y = attrition['DistanceFromHome'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[1, 0])
    axes[1, 0].set(title='Daily Rate against DistancefromHome')

    cmap = sns.cubehelix_palette(start=1.333333333333, light=1, as_cmap=True)
    x = attrition['DailyRate'].values
    y = attrition['JobSatisfaction'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[1, 1])
    axes[1, 1].set(title='Daily Rate against Job satisfaction')

    cmap = sns.cubehelix_palette(start=1.666666666667, light=1, as_cmap=True)
    x = attrition['YearsAtCompany'].values
    y = attrition['JobSatisfaction'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[1, 2])
    axes[1, 2].set(title='Daily Rate against distance')

    cmap = sns.cubehelix_palette(start=2.0, light=1, as_cmap=True)
    x = attrition['YearsAtCompany'].values
    y = attrition['DailyRate'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[2, 0])
    axes[2, 0].set(title='Years at company against Daily Rate')

    cmap = sns.cubehelix_palette(start=2.333333333333, light=1, as_cmap=True)
    x = attrition['RelationshipSatisfaction'].values
    y = attrition['YearsWithCurrManager'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[2, 1])
    axes[2, 1].set(title='Relationship Satisfaction vs years with manager')

    cmap = sns.cubehelix_palette(start=2.666666666667, light=1, as_cmap=True)
    x = attrition['WorkLifeBalance'].values
    y = attrition['JobSatisfaction'].values
    sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[2, 2])
    axes[2, 2].set(title='WorklifeBalance against Satisfaction')

    f.tight_layout()