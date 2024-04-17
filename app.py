import os
import pickle
import joblib
from flask import Flask,render_template,request,redirect,url_for
from check import extract_infos
from werkzeug.utils import secure_filename
from urldetection import detect_malicious_url
from filekeys import get_keys

app= Flask(__name__, static_folder="static")
basePath = r"C:\Users\Juhi\Documents\BE project\FINAL_code_Be_project (2)\User_Uploads"
@app.route("/file", methods=["GET","POST"])
def upload_file():
    if request.method =="POST" :
        f=request.files['file_name']
        f.save(os.path.join(basePath,f.filename))
        g=secure_filename(f.filename) #g=f=user's file name
        
        filePath = os.path.join(basePath,g)

        clf = joblib.load(os.path.join(

        os.path.dirname(os.path.realpath(__file__)),
            'classifier_PKL/classifier.pkl'
        ))
        features = pickle.loads(open(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'classifier_PKL/features.pkl'),
            'rb').read()
        )

        data = extract_infos(filePath)

        pe_features = list(map(lambda x:data[x], features)) #imp ???

        res= clf.predict([pe_features])[0]
        
        prediction = ['malicious', 'legitimate'][res]

        key_data = get_keys(filePath)
        return render_template("file-results.html", fileName=f.filename, prediction = prediction, isMalicious=prediction == 'malicious', key_data=key_data)
    return render_template("upload-file.html")

@app.route("/", methods=["GET"])
def home_page():
    return render_template("home-page.html")

@app.route("/documentation", methods=["GET"])
def Documentation():
    return render_template("Documentation_page.html")

@app.route("/About", methods=["GET"])
def Aboutus():
    return render_template("Aboutus_page.html")


@app.route("/url", methods=["GET","POST"])
def enter_url():
    if request.method =="POST" :
        test_url = request.form['url']
        prediction = detect_malicious_url(test_url)
        print(prediction)
        prediction = "legitimate" if prediction=="good" else "malicious"
        return render_template("file-results.html", urlName=test_url, prediction = prediction, isMalicious=prediction == 'malicious')
    return render_template("upload-url.html")


if __name__=="__main__":
    app.run(debug=True)
