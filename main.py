import pandas
import matplotlib.pyplot as plt

d = pandas.read_csv("data/House Price India.csv")

# print(d.head())
# print(d.info())
#print(d.columns)
# s=d["Price"].describe().reset_index()
# s["Price"]=round(s["Price"],2)
# print(s)
# print(d.isna().sum().sum())
# print(d.duplicated().sum())
# print(d.groupby("condition of the house")["Price"].mean().reset_index().sort_values("Price", ascending=False))

M=d.groupby("condition of the house")["Price"].mean().reset_index().sort_values("Price", ascending=False)
plt.bar(M["condition of the house"], M["Price"])
plt.xlabel("Condition of the House")
plt.ylabel("Average Price")
plt.title("Average House Price by Condition")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

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
#print(x.head())
print("--"*25,"X data","--"*25)
print("X data",x.info())
print("--"*25,"y data","--"*25)
print(y.info())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
t = x_train.join(y_train)

# Fit Linear Regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Print model score
print("Linear Regression R^2 score:", lr.score(x_test, y_test))

new_data = [[42491, 3, 2, 1500, 2000, 1, 0, 0, 3, 7, 1500, 0, 2000, 0, 122004, 52.9, -114.5, 1500, 2000, 2, 10]]
predicted_price = lr.predict(new_data)
print("Predicted Price:", predicted_price[0][0])


#print(t)

# Plotting a heatmap of correlations
import seaborn as sns

plt.figure(figsize=(12, 10))
sns.heatmap(t.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Plotting histograms for each feature
t.hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.show()

