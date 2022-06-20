
import numpy as np # linear algebra
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, log_loss, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE

from data_preprocess import readData
from data_preprocess import analyseData
from data_train import pearsonCorr





if __name__ == '__main__':

    attrition = readData()

    analyseData(attrition)

    pearsonCorr(attrition)

    attrition = attrition.drop(['Attrition_numerical'], axis=1)

    categorical = []

    for col, value in attrition.iteritems():
        if value.dtype == 'object':
            categorical.append(col)


    numerical = attrition.columns.difference(categorical)

    atr_cat = attrition[categorical]
    atr_cat = atr_cat.drop(['Attrition'], axis=1)
    atr_cat = pd.get_dummies(atr_cat)
    atr_cat.head(3)
    atr_num = attrition[numerical]
    atr_fin = pd.concat([atr_num, atr_cat], axis=1)

    target_map = {'Yes': 1, 'No': 0}

    target = attrition["Attrition"].apply(lambda x: target_map[x])

    train, test, target_train, target_val = train_test_split(atr_fin,
                                                             target,
                                                             train_size=0.80,
                                                             random_state=0);

    oversampler = SMOTE(random_state=0)
    smote_train, smote_target = oversampler.fit_resample(train, target_train)

    seed = 0  # We set our random seed to zero for reproducibility
    # Random Forest parameters
    rf_params = {
        'n_jobs': -1,
        'n_estimators': 1000,
        #     'warm_start': True,
        'max_features': 0.3,
        'max_depth': 4,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': seed,
        'verbose': 0
    }

    rf = RandomForestClassifier(**rf_params)
    rf.fit(smote_train, smote_target)
    rf_predictions = rf.predict(test)
    print("Accuracy score: {}".format(accuracy_score(target_val, rf_predictions)))
    print("=" * 80)
    print(classification_report(target_val, rf_predictions))

    trace = go.Scatter(
        y=rf.feature_importances_,
        x=atr_fin.columns.values,
        mode='markers',
        marker=dict(
            sizemode='diameter',
            sizeref=1,
            size=13,
            color=rf.feature_importances_,
            colorscale='Portland',
            showscale=True
        ),
        text=atr_fin.columns.values
    )
    data = [trace]

    layout = go.Layout(
        autosize=True,
        title='Random Forest Feature Importance',
        hovermode='closest',
        xaxis=dict(
            ticklen=5,
            showgrid=False,
            zeroline=False,
            showline=False
        ),
        yaxis=dict(
            title='Feature Importance',
            showgrid=False,
            zeroline=False,
            ticklen=5,
            gridwidth=2
        ),
        showlegend=False
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='scatter')