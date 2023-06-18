from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

app = Flask(__name__)

data = pd.read_csv('DataSet.csv')
train_data = data['message']
train_response = data['response']

vectorizer = CountVectorizer()
train_features = vectorizer.fit_transform(train_data)

model = SVC()
model.fit(train_features, train_response)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global data
    input_message = request.form['message']
    predicted_response = predict_response(input_message)

    new_data = pd.DataFrame({'message': [input_message], 'response': [predicted_response]})
    data = pd.concat([data, new_data], ignore_index=True)
    data.to_csv('Teksty.csv', index=False)

    return render_template('index.html', message=input_message, response=predicted_response)

def predict_response(message):
    features = vectorizer.transform([message])
    response = model.predict(features)
    return response[0]

if __name__ == '__main__':
    app.run()
