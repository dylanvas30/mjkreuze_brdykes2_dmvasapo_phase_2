import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer  # Import SimpleImputer
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

def logisticRegression():
    data = pd.read_csv("clean_GSAF5.xls.csv")
    # drop missing data
    data = data.dropna(subset=['Fatal (Y/N)', 'Age'])
    data['Sex of Victim'] = data['Sex of Victim'].map({'M': 0, 'F': 1})
    # Hremove missing values in predicting varilables
    imputer = SimpleImputer(strategy='median')  # use median to impute missing values
    X_imputed = imputer.fit_transform(data[['Sex of Victim', 'Age']])


    X = X_imputed
    #make fatal yes/no into binary 1 or 0
    data['Fatal (Y/N)'] = data['Fatal (Y/N)'].map({'N': 0, 'Y': 1})
    y = data['Fatal (Y/N)']


    # split tdataset into training and testing groups(80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # creating model
    # createlogistic regression model
    model = LogisticRegression()
    # fit model to the training data
    model.fit(X_train, y_train)


    # Predictions
    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)


    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(conf_matrix)


    #matplot graph stuff
    plt.scatter(data['Age'], data['Fatal (Y/N)'], c=data['Fatal (Y/N)'], cmap='coolwarm')
    plt.xlabel("Age")
    plt.ylabel("Fatal (Y/N)")
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.title("Age vs. Fatal (Y/N)")
    plt.show()


logisticRegression()