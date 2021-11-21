import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from flask_ngrok import run_with_ngrok
app = Flask(__name__)
run_with_ngrok(app)

model = joblib.load(open('crowdfundpredict_new.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        description = request.form['description']
        description = [description]
        string = model.predict(description)
        return render_template('index.html', pred=string)

if __name__ == '__main__':
     app.run()
