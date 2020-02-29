import pandas, datetime
import yfinance as yf
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def predict(new, degree, startDay):
    data = yf.download(stock, startDay, str(datetime.datetime.today()).split(" ")[0])

    y = list(data['Adj Close'].values) + [list(yf.Ticker(stock).history(period="1d", interval='1m')['Open'].values)[-1]]
    xList = [i for i in range(1, len(y) + 1)]
    X = pandas.DataFrame(xList).values.reshape(-1, 1)

    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(X)

    model = LinearRegression()
    model.fit(x_poly, y)

    new = pandas.DataFrame(list(range(len(xList) + 1, len(xList) + new))).values.reshape(-1, 1)
    new_poly = polynomial_features.transform(new)

    new_poly_pred = model.predict(new_poly)

    y_poly_pred = model.predict(x_poly)

    pyplot.plot(X.flatten(), y, c='#000000')
    pyplot.scatter(X.flatten(), y_poly_pred, s=1, c=['#0000ff'])
    pyplot.scatter(new.flatten(), new_poly_pred, c='#ff0000', s=2)
    pyplot.title(stock)


    return new_poly_pred[-1]


def predictDay(degree, new):
    y = list(yf.Ticker(stock).history(period="1d", interval='1m')['Open'].values)
    xList = [i for i in range(1, len(y) + 1)]
    X = pandas.DataFrame(xList).values.reshape(-1, 1)

    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(X)

    model = LinearRegression()
    model.fit(x_poly, y)

    new = pandas.DataFrame(list(range(len(xList) + 1, len(xList) + new))).values.reshape(-1, 1)
    new_poly = polynomial_features.transform(new)

    new_poly_pred = model.predict(new_poly)

    y_poly_pred = model.predict(x_poly)

    '''pyplot.plot(X.flatten(), y, c='#000000')
    pyplot.scatter(X.flatten(), y_poly_pred, s=1, c=['#0000ff'])
    pyplot.scatter(new.flatten(), new_poly_pred, c='#ff0000', s=2)
    pyplot.title(stock)'''


    return new_poly_pred[-1]


with open("boughtstonks.txt") as f:
    cur = f.read().split(" ")

stocks = ["TSLA", "GOOG", "AAPL"]
day = 2
curMoney = int(cur[1])
prevPrice = float(cur[3])


for stock in stocks:
    curPrice = list(yf.Ticker(stock).history(period="1d", interval='1m')['Open'].values)[-1]

    pred = (predict(2, 2, '2020-01-01') + predict(2, 3, '2020-01-01')*2
            + predict(2, 2, '2020-02-01') + predict(2, 3, '2020-02-01')*2 + predictDay(2, 2)*3 + predictDay(3, 2)*3 + curPrice)/13


    print(pred - curPrice)

    pyplot.show()


# avg 5 day prediction for degrees 3 and 2, random times between 1 mo and `qœ6 mo
# idea: compare with overall stock category trend? maybe later :)
