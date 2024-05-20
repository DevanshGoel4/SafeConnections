#this code has 2 parts. first using my data set, we will decide wheather to flag the message or not.
#the second part triggers if the ML decides to flag the message, and a message will be sent
#to the parent. In order for a full stack solution, Flask (or another REST api) will be needed.


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from oauth2client.service_account import ServiceAccountCredentials
import gsheets
from twilio.rest import Client
import time


# CONSTANTS YOU MUST CHANGE
my_json = "YOUR JSON"
account_sid = 'YOUR ACCOUNT SID'
auth_token = 'YOUR TOKEN'
notification_number = 'INTENDED NUMBER FOR TEXTING'

# Load the model and vectorizer from disk
model = joblib.load('human_trafficking_detector.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


def classify_message(message):
    # Preprocess the input message
    message_tfidf = vectorizer.transform([message])

    # Make prediction
    prediction = model.predict(message_tfidf)
    prediction_proba = model.predict_proba(message_tfidf)

    unsure_threshold = 0.45  # Higher value more sensitive

    # Determine classification
    if prediction_proba[0][1] > 1 - unsure_threshold:
        return "trafficking message"
    else:
        return "safe message"


def send_notification():
    # Google Sheets interaction

        client = Client(account_sid, auth_token)
        message = client.messages.create(
            from_='YOUR TWILLIO ACCOUNT NUMBER',
            body=f'Hello, this is SafeConnections. We have reason to suspect that a human trafficker '
                 f'has contacted. Please go to their messages and block recent contacts who '
                 f'appear suspicious.',
            to=notification_number
        )


message = input("WHAT IS THE MESSAGE: ")
classification_result = classify_message(message)
print(classification_result)

if classification_result == "trafficking message":
    send_notification()
