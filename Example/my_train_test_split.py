from Regression import MyLinearRegression,my_train_test_split
X = list(range(10))
y = [i % 2 for i in X]
print(X)
print(y)
X_train, X_test, y_train, y_test = my_train_test_split(X, y, test_size=0.2, random_state=42)

print("Training features:", X_train)
print("Testing features:", X_test)
print("Training labels:", y_train)
print("Testing labels:", y_test)