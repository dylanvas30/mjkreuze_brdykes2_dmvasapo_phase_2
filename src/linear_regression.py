import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def shark_linear_regression():
    shark_attacks = pd.read_csv('clean_GSAF5.xls.csv')

    X = shark_attacks[['Year']]
    y = shark_attacks['Age']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Attacks')
    plt.plot(X_test, y_pred, color='red', label='Attacks by Age')
    plt.title('Age of Victims over Years')
    plt.xlabel('Year')
    plt.ylabel('Age')
    plt.legend()
    plt.show()