# from flask import Flask, render_template, request, jsonify
# import pickle

# app = Flask(__name__)

# # Load the machine learning model
# with open('../human_trafficking_detector.pkl', 'rb') as f:
#     model = pickle.load(f)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/send_message', methods=['POST'])
# def send_message():
#     message = request.form['message']

#     # Pass message to the loaded model for prediction
#     prediction = model.predict([message])[0]

#     return jsonify(prediction=prediction)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and vectorizer from disk
model = joblib.load('../human_trafficking_detector.pkl')
vectorizer = joblib.load('../tfidf_vectorizer.pkl')

# Define the threshold
unsure_threshold = 0.3  # Higher value more sensitive


def classify_message(message):
    # Preprocess the input message
    message_tfidf = vectorizer.transform([message])

    # Make prediction
    prediction_proba = model.predict_proba(message_tfidf)

    # Determine classification
    if prediction_proba[0][1] > 1 - unsure_threshold:
        return "trafficking message"
    else:
        return "safe message"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify_message', methods=['POST'])
def classify_message_route():
    message = request.form['message']
    classification_result = classify_message(message)
    print(classification_result)
    return jsonify(classification_result=classification_result)


if __name__ == '__main__':
    app.run(debug=True)
