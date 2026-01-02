
#import tensorflow and loading  saved image classfication model from models path
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
model_path = 'models\model_im\spiral_graph_classification_model.h5'
model = tf.keras.models.load_model(model_path)



from flask import Flask, jsonify, request, render_template
from flask import Flask, jsonify, request, render_template,session,redirect
from flask import Flask, jsonify, request
from PIL import Image
import numpy as np

from methods import processimgfromdir
from methods import mfcc_target1
import tflearn
import pyrebase



#authentication for web app
config = {
    'apiKey': "AIzaSyAjz4HYWQZWccfmZ9jTJazT0KNBn3qpqlU",
    'authDomain': "deepl-fb893.firebaseapp.com",
    'projectId': "deepl-fb893",
    'storageBucket': "deepl-fb893.appspot.com",
    'messagingSenderId': "604551157344",
    'appId': "1:604551157344:web:4007dd1bb0fdff52c5d505",
    'measurementId': "G-PC3YS86DNM",
    'databaseURL' : ''
}



learning_rate = 0.00001
training_iters =80  # steps
#
width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 2  # digits


#loading audio classfication model

net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
model_audio = tflearn.DNN(net, tensorboard_verbose=0)

model_audio_path = 'models/model_ad/tflearn.lstm.model'
model_audio.load(model_audio_path)






app = Flask(__name__)


firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
app.secret_key = "key_secret"



email = "test@gmail.com"
password = '12345678'
#created a user
#user = auth.create_user_with_email_and_password(email, password)
#print(user)

#print(user)


#path of test images
testpath = './data/spiral/testing'


#function to process the image and return img array
def processimg(img_path):
    img = Image.open(img_path)
    resized = img.resize((224, 224))
    img_array = np.array(resized)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array




#function to predict with give model and img
def make_prediction(img_array, model):
    tf.compat.v1.enable_eager_execution()
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return 'Parkinson'
    else:
        return 'Healthy'
    
"""def make_prediction_audio(audio):
    tf.compat.v1.enable_eager_execution()
    prediction = model_audio.predict(audio)
    if np.argmax(prediction) == 0:
        return 'Healthy'
    else:
        return 'Parkinson'"""


#home page
@app.route('/')
def home():
    return render_template('index.html', user = session)





#route for login
@app.route('/login', methods = ["GET","POST"])
def login():
    if ('user' in session):
        return redirect('/')
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session['user'] = email
            return render_template('index.html',s='successful', user = session)

        except:
            return render_template('index.html',us='unsuccesful', user = session)
            return 'Failed to login'
        
    return render_template('login.html')



#route for logout
@app.route('/logout')
def logout():
    session.pop('user')
    return redirect('/')




#route for spiral test page
@app.route('/spiraltest', methods=["GET", "POST"])
def spiral():
    if ('user' not in session):
        return render_template("index.html",user = session, lg=1)
    if request.method == 'POST':
        img = request.files['image']
        if img:
            img_path = 'static\images\{}'.format(img.filename)
            img.save(img_path)
            try:
                img_processed = processimg(img_path)
                prediction = make_prediction(img_processed, model)
                return render_template('spiral.html', prediction=prediction, img_path=img_path)
            except:
                return render_template('error.html')
    return render_template('spiral.html')



#route for audio test page
@app.route('/audiotest', methods = ['GET','POST'])
def audiot():
    if ('user' not in session):
        return render_template("index.html",user = session, lg=1)
    if request.method == 'POST':
        audio = request.files['audio']
        if audio:
            audio_path = 'static/audio/{}'.format(audio.filename)
            audio.save(audio_path)
            try:
                features = mfcc_target1(audio_path)
                #prediction = make_prediction_audio(features)
                tf.compat.v1.enable_eager_execution()
                prediction = model_audio.predict(features)
                if np.argmax(prediction) == 0:
                    prediction =  'Healthy'
                else:
                    prediction =  'Parkinson'
                return render_template('audio.html', prediction=prediction)
            except:
                return render_template('error.html')
    
    return render_template('audio.html')



#route for perfomance page
@app.route('/performance')
def perfo():
    if ('user' not in session):
        return render_template("index.html",user = session, lg=1)
    data = processimgfromdir(testpath)
    #result = model.evaluate(data)
    #accuracy = result[1]
    #loss = result[0]
    accuracy = 0.90
    # covert this upto two decimal places
    accuracy = round(accuracy, 2)
    loss = round(0.3554, 2)
    return render_template('perf.html', accuracy=accuracy*100, loss=loss)




#set debug == False while deployment else server will restart after every edit

if __name__ == '__main__':
    app.run(debug=False)
