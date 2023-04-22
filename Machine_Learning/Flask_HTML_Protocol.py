#!/usr/bin/env python
# coding: utf-8

# In[37]:


from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def greeting():
    return render_template('form.html')

# @app.route('/data/', methods = ['POST', 'GET'])
# def data():
#     form_data = request.form
#     if request.method == 'GET':
#         return str(list(form_data.values()))
#     if request.method == 'POST':
#         form_data = request.form
#         return render_template('data.html',form_data = form_data)

# @app.route('/model')
# def model():
#     loaded_model = pickle.load(open('Capstone_RFC_model.sav','rb'))
#     a = [1,1,0,0,0,1,1,1,0,1,0]
#     a = np.expand_dims(a,0)
#     result = str(loaded_model.predict(a))
#     return result


# @app.route("/calculate-estimate", methods=["GET"])
# def calculate_estimate():
#     estimate = request.args.get("estimate")
#     return estimate

# @app.route("/fav_language", methods=["GET"])
# def fav_language():
#     estimate = request.args.get("fav_language")
#     return estimate

@app.route("/result", methods=['POST'])
def predict(): 
    form_data = request.form
    x = list(form_data.values())
    x.pop(0)
    y = ['Age', 
         'Diarrhea', 
         'Difficulty in Breathing', 
         'Dry Cough', 
         'Fever', 
         'Nasal', 
         'Pains', 
         'Runny Nose', 
         'Sore Throat', 
         'Tireness']
    z = [int(x[0])]
    for i in range(len(y)):
        if y[i] not in x:
            z.insert(i+1, 0)
        else :
            y[i]=1
            z.append(y[i])
    print(z)
    loaded_model = pickle.load(open('Capstone_RFC_model.sav','rb'))
    a = np.expand_dims(z,0)
    result = str(loaded_model.predict(a))
    print(result)
    return result


if __name__=='__main__':
    app.run()


# In[ ]:




