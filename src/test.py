import matplotlib
matplotlib.use('Agg')  # Explicitly set the backend to 'Agg' (Agg is a non-interactive backend)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def speciesAttacks(shark_attacks):
    plt.figure(figsize=(12, 6))
    shark_attacks = shark_attacks[shark_attacks['Species'] != 'Unknown']
    sns.countplot(x='Species', data=shark_attacks, order=shark_attacks['Species'].value_counts().index[:10])
    plt.title('Distribution of Shark Attacks by Species')
    plt.xticks(rotation=45)
    plt.tight_layout()

def countryAttacks(shark_attacks):
    plt.figure(figsize=(12, 8))
    sns.countplot(x='Country', data=shark_attacks, order=shark_attacks['Country'].value_counts().index[:10])
    plt.title('Top 10 Countries with Most Shark Attacks')
    plt.xticks(rotation=45)

def activityAttacks(shark_attacks):
    top_activities = shark_attacks['Activity'].value_counts().index[:10]
    plt.figure(figsize=(12, 8))
    sns.countplot(y='Activity', data=shark_attacks, order=top_activities)
    plt.title('Distribution of Shark Attacks by Activity')

def fatalAttacks(shark_attacks):
    plt.figure(figsize=(8, 8))
    shark_attacks['Fatal (Y/N)'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('Proportion of Fatal and Non-Fatal Shark Attacks')

def monthlyDistribution(shark_attacks):
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Month', data=shark_attacks, order=range(1, 13))
    plt.title('Monthly Distribution of Shark Attacks')

def ageDistribution(shark_attacks):
    plt.figure(figsize=(10, 6))
    sns.histplot(x='Age', data=shark_attacks, bins=30, kde=True)
    plt.title('Age Distribution of Shark Attack Victims')
    plt.xlabel('Age')
    plt.ylabel('Number of Attacks')
    plt.xlim(0, 100)

def timeOfDayDistribution(shark_attacks):
    shark_attacks['Hour'] = shark_attacks['Time'].str[:2]
    shark_attacks['Hour'] = shark_attacks['Hour'].str.replace('h', '')
    shark_attacks['Hour'] = shark_attacks['Hour'].astype(int)
    plt.figure(figsize=(10, 6))
    shark_attacks_sorted = shark_attacks.sort_values(by='Hour')
    sns.histplot(x='Hour', data=shark_attacks_sorted, bins=24, kde=False, discrete=True)
    plt.title('Time of Day Distribution of Shark Attacks')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Attacks')
    plt.xticks(rotation=45)

def shark_linear_regression(shark_attacks):
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

def shark_naive_bias(shark_attacks):
    shark_attacks['Fatal (Y/N)'] = shark_attacks['Fatal (Y/N)'].fillna('N')
    shark_attacks['target'] = shark_attacks['Fatal (Y/N)'].apply(lambda x: 1 if x == 'Y' else 0)
    shark_attacks['Activity'] = shark_attacks['Activity'].fillna('').astype(str)

    X_train, X_test, y_train, y_test = train_test_split(shark_attacks['Activity'], shark_attacks['target'], test_size=0.2, random_state=42)

    clf = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('nb', MultinomialNB())
    ])

    clf.fit(X_train.values.ravel(), y_train)

    y_pred = clf.predict(X_test.values.ravel())

    plt.figure(figsize=(8, 6))
    pd.Series(y_pred).value_counts().plot(kind='bar', color=['blue', 'red'])
    plt.title('Fatality Prediction')
    plt.xlabel('Predicted Fatal or Non Fatal')
    plt.ylabel('Attacks')
    plt.xticks([0, 1], ['Non Fatal', 'Fatal'], rotation=0)

    return clf

# Test the function
#speciesAttacks(shark_attacks)
#countryAttacks(shark_attacks)
#activityAttacks(shark_attacks)
#fatalAttacks(shark_attacks)
#monthlyDistribution(shark_attacks)
#ageDistribution(shark_attacks)
#timeOfDayDistribution(shark_attacks)
#shark_linear_regression(shark_attacks)
#shark_naive_bias(shark_attacks)