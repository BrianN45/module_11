# CS2023 - Lab11

_author_ = "Brian Nguyen"
_credits_ = ["N/A"]
_email_ = "nguyeb2@mail.uc.edu"

'''
Multiple Linear Regression with All Datasets
R2 Score: 0.6008983115964333
MSE Score: 0.5350149774449119


MedInc
R2 Score: 0.4630810035698605
MSE Score: 0.7197656965919479


HouseAge
R2 Score: 0.013185632224592903
MSE Score: 1.3228720450408296


AveRooms
R2 Score: 0.024105074271276283
MSE Score: 1.3082340086454287


AveBedrms
R2 Score: -0.0011266270315772875
MSE Score: 1.3420583158224824


Population
R2 Score: 8.471986797708997e-05
MSE Score: 1.3404344471369465


AveOccup
R2 Score: -0.00018326453581640756
MSE Score: 1.340793693098357


Latitude
R2 Score: 0.020368890210145207
MSE Score: 1.3132425427841639


Longitude
R2 Score: 0.0014837207852690382
MSE Score: 1.3385590192298276

It seems that most of the columns have little to no effect on house prices with population being affecting the least.
The only column that seems to have the noticeable effect is the median income, which has an R2 score of 0.46 and an MSE
of 0.72. However, despite the other columns having very small R2 scores, combining all of them into one linear regression
model yields an R2 score of 0.60 and an MSE score of 0.54, making it the most accurate of them all.
'''

# Script to Visualize the Expected vs.
# Predicted Prices using Multiple Linear
# Regression Housing Price Estimator

import pandas as pd
from sklearn.datasets import fetch_california_housing

def printScores(r2, mse):
    print("R2 Score: {0}\nMSE Score: {1}".format(r2, mse))

cali = fetch_california_housing()
cali_df = pd.DataFrame(cali.data,
                       columns=cali.feature_names)
cali_df['MedHouseValue'] = pd.Series(cali.target)
print(cali_df['MedHouseValue'].shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cali.data, cali.target, random_state=11)
from sklearn.linear_model import LinearRegression
from sklearn import metrics

mu_regress = LinearRegression()
mu_regress.fit(X=X_train, y=y_train)
predicted = mu_regress.predict(X_test)
expected = y_test
z = zip(predicted[::1000], expected[::1000])
df = pd.DataFrame()
df['Expected'] = pd.Series(expected)
df['Predicted'] = pd.Series(predicted)

R2 = metrics.r2_score(expected, predicted)
MSE = metrics.mean_squared_error(expected, predicted)

print('Multiple Linear Regression with All Datasets')
printScores(R2, MSE)
print('\n')

for i in cali.feature_names:
    X_train, X_test, y_train, y_test = train_test_split(cali_df[i].values.reshape(-1, 1), cali.target, random_state=11)
    mu_regress.fit(X=X_train, y=y_train)
    predicted = mu_regress.predict(X_test)
    expected = y_test
    z = zip(predicted[::1000], expected[::1000])
    df = pd.DataFrame()
    df['Expected'] = pd.Series(expected)
    df['Predicted'] = pd.Series(predicted)

    R2 = metrics.r2_score(expected, predicted)
    MSE = metrics.mean_squared_error(expected, predicted)

    print(i)
    printScores(metrics.r2_score(expected, predicted), metrics.mean_squared_error(expected, predicted))
    print('\n')