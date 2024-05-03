import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def shark_naive_bias():
    shark_attacks = pd.read_csv("clean_GSAF5.xls.csv")
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
    plt.show()

    return clf