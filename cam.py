import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from flask_ngrok import run_with_ngrok
app = Flask(__name__,template_folder='template')
run_with_ngrok(app)

model = joblib.load(open('crowdfundpredict_new.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('crowdfundpredict.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        description = request.form['description']
        description = [description]
        string = model.predict(description)
        return render_template('crowdfundpredict.html', pred=string)

if __name__ == '__main__':
     app.run()
