import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv('sample_data.csv')
data = data.dropna()

def trainmodel(data):
    # Split data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Vectorize text data
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data['text'])
    X_test = vectorizer.transform(test_data['text'])

    # Save vectorizer and processed data
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(X_train, 'X_train.pkl')
    joblib.dump(X_test, 'X_test.pkl')


    # Train a Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train, train_data['label'])

    # Save the trained model
    joblib.dump(model, 'model.joblib')

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(test_data['label'], y_pred)
    return accuracy

if __name__ == "__main__":
    data = pd.read_csv('sample_data.csv')
    data = data.dropna()
    trainmodel(data)