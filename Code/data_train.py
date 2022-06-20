import plotly.offline as py
import plotly.graph_objs as go



def pearsonCorr(attrition):

    target_map = {'Yes': 1, 'No': 0}

    attrition["Attrition_numerical"] = attrition["Attrition"].apply(lambda x: target_map[x])

    numerical = [u'Age', u'DailyRate', u'DistanceFromHome',
                 u'Education', u'EmployeeNumber', u'EnvironmentSatisfaction',
                 u'HourlyRate', u'JobInvolvement', u'JobLevel', u'JobSatisfaction',
                 u'MonthlyIncome', u'MonthlyRate', u'NumCompaniesWorked',
                 u'PercentSalaryHike', u'PerformanceRating', u'RelationshipSatisfaction',
                 u'StockOptionLevel', u'TotalWorkingYears',
                 u'TrainingTimesLastYear', u'WorkLifeBalance', u'YearsAtCompany',
                 u'YearsInCurrentRole', u'YearsSinceLastPromotion', u'YearsWithCurrManager']
    data = [
        go.Heatmap(
            z = attrition[numerical].astype(float).corr().values,  x = attrition[numerical].columns.values, y = attrition[numerical].columns.values,
            colorscale = 'Viridis',
            reversescale = False,
            opacity=1.0
        )
    ]

    layout = go.Layout(
        title='Pearson Korelasyonu',
        xaxis=dict(ticks='', nticks=36),
        yaxis=dict(ticks=''),
        width=900, height=700,

    )

    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename='labelled-map')