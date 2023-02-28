import numpy as np
from flask import Flask, request, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('diabetes_predictor.pkl','rb'))

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    return render_template('index.html', prediction_text="If a 1 then you have a diabetes, if a 0 then no diabetes. You are a {}".format(output))
if __name__ == "__main__":
    app.run(port=5000, debug=True)
