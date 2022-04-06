from flask import Flask , render_template , request
import pickle
import numpy as np


app = Flask(__name__)

filename = 'kidney_prediction_pickle'
model = pickle.load(open(filename,'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    sg = float(request.form['sg'])
    al  = float(request.form['al'])
    rbc = float(request.form['rbc'])
    pc = float(request.form['pc'])
    bgr = float(request.form['bgr'])
    sc = float(request.form['sc'])
    hemo = float(request.form['hemo'])
    htn = float(request.form['htn'])
    appet = float(request.form['appet'])
    pe = float(request.form['pe'])

    data = np.array([[sg,al,rbc,pc,bgr,sc,hemo,htn,appet,pe]])
    my_prediction = model.predict(data)

    return render_template('index.html',
    sg = str(sg),
    al = str(al),
    rbc = str(rbc),
    pc = str(pc),
    bgr = str(bgr),
    sc = str(sc),
    hemo = str(hemo),
    htn = str(htn),
    appet = str(appet),
    pe = str(pe),
    prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)