import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

encoder = CountVectorizer(binary=True)

def load_file(name):
    df = pd.read_csv(name,
                header=None,
                 names=['Sentiment','Headline'], encoding="ISO-8859-1")
    return df

def preprocess(df):
    df = df.drop_duplicates()
    df = df[["Headline", "Sentiment"]]

    df.loc[df['Sentiment'] == 'neutral', 'Sentiment'] = 0
    df.loc[df['Sentiment'] == 'positive', 'Sentiment'] = 1
    df.loc[df['Sentiment'] == 'negative', 'Sentiment'] = -1
    df['Sentiment'] = pd.to_numeric(df['Sentiment'])

    X = df.Headline
    y= df.Sentiment
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test

def build_model_and_save(X_train, X_test, y_train, y_test):
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    joblib.dump(model, "setnti_predict.pkl")
    # model = joblib.load('setnti_predict.pkl')
    # print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))


def load_New_file(name):
    df = pd.read_csv(name,
                header=None,
                 names=['Headline'])
    df['Sentiment'] = 0
    df = df.drop_duplicates()
    return df

def preprocess_New_file(df):

    X_test = df.Headline
    y_test = df.Sentiment

    X_test = encoder.transform(X_test)
    return X_test,y_test

def load_model_predict(X_test,y_test):
    model = joblib.load('setnti_predict.pkl')
    y_test = model.predict(X_test)
    joblib.dump(model, "setnti_predict.pkl")
    # print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    return y_test

if __name__ == '__main__':
    df = load_file('./csv/Sentiment Analysis for Financial News.csv')
    X_train, X_test, y_train, y_test = preprocess(df)
    build_model_and_save(X_train, X_test, y_train, y_test)

    scraped_data = load_New_file("csv/Data.csv")
    X_test,y_test = preprocess_New_file(scraped_data)
    y_test = load_model_predict(X_test,y_test)
    scraped_data['Sentiment'] = y_test
    print(scraped_data.info())


