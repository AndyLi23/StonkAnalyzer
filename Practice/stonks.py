import pandas
import yfinance as yf
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


stock = 'AMZN'

data = yf.download(stock, '2010-01-01', '2020-02-10')

y = list(data['Adj Close'].values)
xList = [i for i in range(1, len(y) + 1)]
X = pandas.DataFrame(xList).values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


polynomial_features= PolynomialFeatures(degree=4)
x_poly = polynomial_features.fit_transform(X_train)

model = LinearRegression()
model.fit(x_poly, y_train)

new = pandas.DataFrame(list(range(len(xList)+1, len(xList)+500))).values.reshape(-1, 1)
new_poly = polynomial_features.transform(new)

new_poly_pred = model.predict(new_poly)

y_poly_pred = model.predict(x_poly)


pyplot.plot(X.flatten(), y, c='#000000')
pyplot.scatter(X_train.flatten(), y_poly_pred, s=1, c=['#0000ff'])
pyplot.plot(new.flatten(), new_poly_pred, c='#0000ff', linewidth = 2.5)
pyplot.title(stock)

pyplot.show()


