
from Regression import MyLinearRegression,my_train_test_split

import pandas

d = pandas.read_csv("data/House Price India.csv")
x = d[['Date',
       'number of bedrooms',
       'number of bathrooms',
       'living area',
       'lot area',
       'number of floors',
       'waterfront present',
       'number of views',
       'condition of the house',
       'grade of the house',
       'Area of the house(excluding basement)',
       'Area of the basement',
       'Built Year',
       'Renovation Year',
       'Postal Code',
       'Lattitude',
       'Longitude',
       'living_area_renov',
       'lot_area_renov',
       'Number of schools nearby',
       'Distance from the airport']]
y = d[["Price"]]

print("--"*25,"X data","--"*25)
print(x.info())
print("--"*25,"y data","--"*25)
print(y.info())
x_train, x_test, y_train, y_test = my_train_test_split(x, y, test_size=0.2, random_state=42)


mlr = MyLinearRegression()
mlr.fit(x_train, y_train)

print("Linear Regression R^2 score:", mlr.score(x_test, y_test))

new_data = [[42491, 3, 2, 1500, 2000, 1, 0, 0, 3, 7, 1500, 0, 2000, 0, 122004, 52.9, -114.5, 1500, 2000, 2, 10]]
predicted_price = mlr.predict(new_data)
print("Predicted Price:", predicted_price[0])