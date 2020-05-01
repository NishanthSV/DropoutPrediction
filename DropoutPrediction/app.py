import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('dropoutprediction.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('dropIndex.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(int_features)
    prediction = model.predict(final_features)
    output =prediction[0]
    if output==True:
        return render_template('dropIndex.html', prediction_text='You have a high probability of getting dropped out!')
    else:
        return render_template('dropIndex.html', prediction_text='You have a low probability of getting dropped out!')

    

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)