import pandas, datetime
import yfinance as yf
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

stock = 'TSLA'


def predict(new, degree, startDay):
    data = yf.download(stock, startDay, str(datetime.datetime.today()).split(" ")[0])

    y = list(data['Adj Close'].values)
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
    pyplot.plot(new.flatten(), new_poly_pred, c='#0000ff', linewidth=2.5)
    pyplot.title(stock)

    pyplot.show()

    return new_poly_pred[-1]


print(predict(5, 3, '2020-01-01'))

# avg 5 day prediction for degrees 3 and 2, random times between 1 mo and 6 mo
# idea: compare with overall stock category trend? maybe later :)
