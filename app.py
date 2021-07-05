from flask import *
from flask_cors import CORS,cross_origin
import warnings
import os
import dash
import plotly.express as px
from flask import Flask, render_template #this has changed
import plotly.graph_objs as go
import numpy as np
import dash_core_components as dcc
import uuid
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.preprocessing import image
import pickle
import matplotlib.pyplot as plt
import pandas as pd 
import requests
from bs4 import BeautifulSoup
url = "https://www.mygov.in/covid-19/"
path_Model = "./static/model/vgg19_pneumonia.h5"
model = tf.keras.models.load_model(path_Model,compile = True)
# model.load_weights("./static/model/vgg19_pneumonia_weights.hdf5")
path_model_Classifier = "./static/model/neuralNet_Covid_Classifier.sav"
# path_Model_BrainTumor = "./static/model/inception_braintumor.h5"
path_Model_BrainTumor = "./static/model/inception_braintumor1.h5"
path_Model_Covid_CT = "./static/model/vgg_ct.h5"
path_Model_Covid_CXRAY = "./static/model/vgg19_covid_chest.h5"
model_Classifer = pickle.load(open(path_model_Classifier,"rb"))
modelBrainTumor = tf.keras.models.load_model(path_Model_BrainTumor,compile = True)
modelCovid_CT = tf.keras.models.load_model(path_Model_Covid_CT,compile = True)
modelCovid_CXRAY = tf.keras.models.load_model(path_Model_Covid_CXRAY,compile = True)
# model_Classifer.load_weights("./static/model/neuralNet_Covid_Classifier_Weight.hdf5")

warnings.filterwarnings("ignore")
UPLOAD_FOLDER = './static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
examples = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'examples'))
@app.route('/')
def home():
    context = web_scrap_and_return_data()
    return render_template('index.html', page='home',context = context)
@app.route('/about')
def about():
    return render_template('about.html', page='about')
@app.route('/detectB')
def detectB():
    return render_template('detectB.html', page='detectB')
@app.route('/detectP')
def detectP():
    return render_template('detectP.html', page='detectP')
@app.route('/detectC')
def detectC():
    return render_template('detectC1.html', page='detectC')
@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html', page='symptoms')
@app.route('/prevention')
def prevention():
    return render_template('prevention.html', page='prevention')
@app.route('/predict', methods=['GET', 'POST'])
def predict1():
    if request.method == 'POST':
        if 'xrayimage' not in request.files:
            return redirect(request.url)
            
        file = request.files['xrayimage']

        if file.filename == '':
            return redirect(request.url)

        if file and check_file_ext(file.filename):
            filename = secure_filename(file.filename)
            file_ext = os.path.splitext(filename)[1]

            filename = str(uuid.uuid4()) + file_ext
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)

            res,prob = predict(filepath)

            return render_template('predict.html', image=filename, result=res, prob = prob)
    else:
        ex_file = request.args.get('example', default=examples[0], type=str)

        if ex_file in examples:
            pass
            # res = predict(os.path.join(app.config['UPLOAD_FOLDER'], 'examples', ex_file)) 
            # return render_template('predict.html', image=os.path.join('examples', ex_file), result=res)
         
    return redirect('/')
@app.route('/predictCovidTesting', methods=['GET', 'POST'])
def predictCovidTesting():
    content = {'state': 'Andhra Pradesh', 'gender': 'male', 'age': '33', 'fever': 'Yes', 'cough': 'Yes', 'fatigue': 'Yes', 'ncongestion': 'Yes', 'pains': 'Yes', 'sbreadth': 'Yes', 'vomiting': 'Yes', 'Diarrhea': 'Yes', 'chills': 'Yes', 'rnose': 'Yes', 'sthroat': 'Yes', 'Headache': 'Yes', 'typeimage': 'XRAY', 'lives_in_affected_area': 1}
    return render_template('predictCovidTesting.html', image="filename", result="REEESSIII", prob = 1, image_inp = "CT-SCAN", content = [content])
@app.route('/predictCovid', methods=['GET', 'POST'])
def predictCovid():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
            
        file = request.files['image']

        if file.filename == '':
            return redirect(request.url)

        if file and check_file_ext(file.filename):
            filename = secure_filename(file.filename)
            file_ext = os.path.splitext(filename)[1]

            filename = str(uuid.uuid4()) + file_ext
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            content ={ 
                "state" : request.form.get("state"),
                "gender" : request.form.get("gender"),
                "age" : request.form.get("age"),
                "fever" : request.form.get("fever"),
                "cough" : request.form.get("cough"),
                "fatigue" : request.form.get("fatigue"),
                "ncongestion" : request.form.get("ncongestion"),
                "pains" : request.form.get("pains"),
                "sbreadth" : request.form.get("sbreadth"),
                "vomiting" : request.form.get("vomiting"),
                "Diarrhea" : request.form.get("Diarrhea"),
                "chills" : request.form.get("chills"),
                "rnose" : request.form.get("rnose"),
                "sthroat" : request.form.get("sthroat"),
                "Headache" : request.form.get("Headache"),
                "typeimage" : request.form.get("typeimage"),
                "lives_in_affected_area" : lives_Cal(request.form.get("state"))}
            # print(content)
            cc = prepare_C(content)
            # print(cc)
            result = (model_Classifer.predict_proba([cc]))
            # result = round(result[0][1]*100,2) 
            labels = ['NORMAL','COVID-19']
            # return make_response(jsonify(return_content), 200)
            res,prob = predictCovidddd(filepath,content['typeimage'])
            lab = np.argmax(result[0])
            return render_template('predictCovid.html', image=filename, result=res, prob = prob, image_inp = "X-RAY Scan" if content['typeimage'] == "XRAY" else "CT-SCAN",
                result2= labels[lab], prob2 = result[0][lab],content = [content])
    else:
        ex_file = request.args.get('example', default=examples[0], type=str)

        if ex_file in examples:
            pass
            # res = predict(os.path.join(app.config['UPLOAD_FOLDER'], 'examples', ex_file)) 
            # return render_template('predict.html', image=os.path.join('examples', ex_file), result=res)
         
    return redirect('/')
def predictCovidddd(filename,typeimage):
    disease_class=['Covid-19','NORMAL']
    if typeimage == "XRAY":
        custom = modelCovid_CXRAY.predict(prepare(filename))
        print("XX")
    else:
        custom = modelCovid_CT.predict(prepare(filename))
        print("CT")
    a=custom[0]       
    # strr = disease_class[1 if a>0.6 else 0]+" with Probability of Pneumonia Being : "+str(a)
    # return strr
    cla = np.argmax(a)
    print(a)
    return disease_class[cla], str(a[cla])
# Other functions
def prepare_C(content):
    ccc = []
    ccc.append(int(content['age']))
    ccc.append(1 if content['gender']=='male' else 0)
    ccc.append(1 if content['fever']=='Yes' else 0)
    ccc.append(int(1 if content['cough']=='Yes' else 0))
    ccc.append(int(1 if content['fatigue']=='Yes' else 0))
    ccc.append(int(1 if content['pains']=='Yes' else 0))
    ccc.append(int(1 if content['ncongestion']=='Yes' else 0))
    ccc.append(int(1 if content['sbreadth']=='Yes' else 0))
    ccc.append(int(1 if content['rnose']=='Yes' else 0))
    ccc.append(int(1 if content['sthroat']=='Yes' else 0))
    ccc.append(int(1 if content['Diarrhea']=='Yes' else 0))
    ccc.append(int(1 if content['chills']=='Yes' else 0))
    ccc.append(int(1 if content['Headache']=='Yes' else 0))
    ccc.append(int(1 if content['vomiting']=='Yes' else 0))
    ccc.append(int(content['lives_in_affected_area']))
    return ccc
def lives_Cal(state):
    return 1

def prepare(path):
    show_img=image.load_img(path, grayscale=False, target_size=(224, 224))
    x = image.img_to_array(show_img)
    x = np.expand_dims(x, axis = 0)
    x /= 255
    plt.imshow(show_img)
    plt.show()
    return x

def prepareForB(path):
    show_img=image.load_img(path, grayscale=True, target_size=(150, 150))
    x = image.img_to_array(show_img)
    x = np.expand_dims(x, axis = 0)
    x /= 255
    plt.imshow(show_img)
    plt.show()
    return x

def my_figure():
    india_states = json.load(open("static/states_india.geojson", "r"))
    state_id_map = {}

    for feature in india_states["features"]:
        feature["id"] = feature["properties"]["state_code"]
        state_id_map[feature["properties"]["st_nm"]] = feature["id"]
    data_scraped = web_scrap_and_return_data()
    df = pd.DataFrame.from_dict(data_scraped['data'])
    df['state_name'] = df['state_name'].str.replace('Andaman and Nicobar','Andaman & Nicobar Island')
    df['state_name'] = df['state_name'].str.replace('Arunachal Pradesh','Arunanchal Pradesh')
    # Dadara & Nagar Havelli
    # Daman & Diu
    df['state_name'] = df['state_name'].str.replace('Dadra and Nagar Haveli and Daman and Diu','Daman & Diu')
    df['state_name'] = df['state_name'].str.replace('Delhi','NCT of Delhi')
    df['state_name'] = df['state_name'].str.replace('Jammu and Kashmir','Jammu & Kashmir')
    df = df[df['state_name']!='Ladakh']
    df['state_name'] = df['state_name'].str.replace('Telengana','Telangana')
    df["id"] = df["state_name"].apply(lambda x: state_id_map[x])
    cols_dd = ["comfirmed_number_total", "active_number", "discharged_number",'total_vaccinated','death_number']
    visible = np.array(cols_dd)
    traces = []
    buttons = []
    colors = {'comfirmed_number_total':'Viridis','active_number':'Reds','discharged_number':'Greens','total_vaccinated':'Bluered','death_number':'Greys'}
    for value in cols_dd:
        traces.append(go.Choropleth(
            locations=df["id"],
            geojson=india_states,
            z=df[value],
            colorbar_title="Numbers",
            hovertext=df["state_name"],
            hovertemplate='State: %{hovertext} '+'<br>'+value+' : %{z}',
#           hoverlabel=df['state_name']],
            colorscale = colors[value],
            zauto=True,
            name = value,
            # showscale = False,
            # showlegend = False,
            visible= True if value==cols_dd[0] else False))

        buttons.append(dict(label=value,method="update",
                        args=[{"visible":list(visible==value)},
                              {"title":f"<b>{value}</b>"}]))

        updatemenus = [{"active":0,
                "buttons":buttons,
                "direction":"down",
            "pad":{"r": 10, "t": 10},
            'showactive':True,
            'x':0.4,
            'xanchor':"left",
            'y':1.2,
            'yanchor':"top"
               }]

    fig = go.Figure(data=traces,
                layout=dict(updatemenus=updatemenus))
    fig.update_layout(
        annotations=[
        dict(text="Select Plot Type:", showarrow=False,
        x=0, y=1.15, yref="paper", align="left")
    ]
    )
    first_title = cols_dd[0]
    fig.update_layout(title=f"<b>{first_title}</b>",title_x=0.6,title_y=1)
    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_layout(
        title=dict(
            xanchor='center',
            x=0.5,
            yref='paper',
            yanchor='bottom',
            y=1,
            pad={'b': 10}
        ),
        margin={'r': 0, 't': 10, 'l': 0, 'b': 0},
        height=550,
        width=600
    )
    fig.update_traces(colorbar_thickness=4, selector=dict(type='choropleth'))
    fig.update_traces(colorbar_len=0.6, selector=dict(type='choropleth'))
    fig.update_traces(selected_marker_opacity=0.7, selector=dict(type='choropleth'))
    return fig

def predict(filename):
    disease_class=['NORMAL','Pnuemonia']
    custom = model.predict(prepare(filename))
    a=custom[0]       
    # strr = disease_class[1 if a>0.6 else 0]+" with Probability of Pneumonia Being : "+str(a)
    # return strr
    return disease_class[1 if a>0.6 else 0], str(a)

def check_file_ext(filename):
    valid_ext = set(['png', 'jpg', 'jpeg', 'jfif'])
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in valid_ext

@app.route('/predictBrainTumor', methods=['GET', 'POST'])
def predict1B():
    if request.method == 'POST':
        if 'mriimage' not in request.files:
            return redirect(request.url)
            
        file = request.files['mriimage']

        if file.filename == '':
            return redirect(request.url)

        if file and check_file_ext(file.filename):
            filename = secure_filename(file.filename)
            file_ext = os.path.splitext(filename)[1]

            filename = str(uuid.uuid4()) + file_ext
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)

            res,prob = predictBrain(filepath)

            return render_template('predictBB.html', image=filename, result=res, prob = prob)
    else:
        ex_file = request.args.get('example', default=examples[0], type=str)

        if ex_file in examples:
            pass
            # res = predict(os.path.join(app.config['UPLOAD_FOLDER'], 'examples', ex_file)) 
            # return render_template('predict.html', image=os.path.join('examples', ex_file), result=res)
         
    return redirect('/')
def web_scrap_and_return_data():
    html = requests.get(url).content
    soup = BeautifulSoup(html,'html.parser')
    data_combined = {}
    summary = soup.find_all( class_ = "icount" )
    data_combined = {'Total Cases':'','Active Cases':'','Discharged/Recovered':'','Deaths/Diseased':''}
    j = 0
    for i in data_combined:
        data_combined[i] = int(summary[j].text.replace(',',''))
        j = j+1
    summary_2 = soup.find_all( class_ = "increase_block" )
    data_ = {'Increment/Decrement in Total Cases':{},'Increment/Decrement in Active Cases':{},'Increment/Decrement in Discharged/Recovered':{},'Increment/Decrement in Deaths/Diseased':{}}
    j = 0
    for i in data_:
        dd = {}
        number = int(summary_2[j].text.replace(',',''))
        dd['number'] = number
        dd['type'] = 'increment' if 'red' in str(summary_2[j]) else 'decrement'
        data_[i] = dd
        j = j+1
    data_combined['data_increment'] = data_
    data = []
    aa = soup.find_all( class_ = "views-row")
    for i in range(36):
        dd = {}
        state_name = aa[i].find(class_ ='st_name').text
        dd['state_name'] = state_name
        comfirmed_number_total = int(aa[i].find(class_ ='st_number').text.replace(',',''))
        dd['comfirmed_number_total'] = comfirmed_number_total
        active_number = int(aa[i].find(class_ ='tick-active').text.split(" ")[1].replace(',',''))
        dd['active_number'] = active_number
        discharged_number = int(aa[i].find(class_ ='tick-discharged').text.split(" ")[1].replace(',',''))
        dd['discharged_number'] = discharged_number
        death_number = int(aa[i].find(class_ ='tick-death').text.split(" ")[1].replace(',',''))
        dd['death_number'] = death_number
        ticktotalvaccine = int(aa[i].find(class_ = 'tick-total-vaccine').text.split(" ")[1].replace(',',''))
        dd['total_vaccinated'] = ticktotalvaccine
        data.append(dd)
        #         data[state_name] = {'Total Confirmed': comfirmed_number_total, 'Active Cases': active_number,'Total Discharge Number':discharged_number,'Death Number':death_number,'Total Vaccination': ticktotalvaccine}
    data_combined['data'] = data
    return data_combined

def predictBrain(filename):
    disease_class=['Glioma Brain Tumor','Meningioma Brain Tumour', 'No Brain Tumour','Pituitary Brain Tumour']
    custom = modelBrainTumor.predict(prepareForB(filename))
    clas = np.argmax(custom[0])
    print(custom)
    return disease_class[clas], custom[0][clas]

@app.route('/api')
@cross_origin()
def hello_world():
    d = {}
    d['Query'] = str(request.args['query'])
    return jsonify(d)
@app.route('/api/submit', methods=["POST"])
@cross_origin()
def submit_api():
    #return_content = user_handler.signup(content)
    if 'file' not in request.files:
        # flash('No file part')
        # return redirect(request.url)
        content = request.get_json(silent=True)
        return_content = content
        return make_response(jsonify(return_content), 200)
    file = request.files['file']
    if file.filename == '':
        # flash('No selected file')
        content = request.get_json(silent=True) 
    if file:
        # filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))
        content = {"STATUS":"FILE SAVED"}
    # resnet_chest = load_model('models/resnet_chest.h5')
    # vgg_chest = load_model('models/vgg_chest.h5')
    # inception_chest = load_model('models/inceptionv3_chest.h5')
    # xception_chest = load_model('models/xception_chest.h5')

    # image = cv2.imread('./flask app/assets/images/upload_chest.jpg') # read file 
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
    # image = cv2.resize(image,(224,224))
    # image = np.array(image) / 255
    # image = np.expand_dims(image, axis=0)

    # resnet_pred = resnet_chest.predict(image)
    # probability = resnet_pred[0]
    # print("Resnet Predictions:")
    # if probability[0] > 0.5:
    #     resnet_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
    # else:
    #     resnet_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
    # print(resnet_chest_pred)

    # vgg_pred = vgg_chest.predict(image)
    # probability = vgg_pred[0]
    # print("VGG Predictions:")
    # if probability[0] > 0.5:
    #     vgg_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
    # else:
    #     vgg_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
    # print(vgg_chest_pred)

    # inception_pred = inception_chest.predict(image)
    # probability = inception_pred[0]
    # print("Inception Predictions:")
    # if probability[0] > 0.5:
    #     inception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
    # else:
    #     inception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
    # print(inception_chest_pred)

    # xception_pred = xception_chest.predict(image)
    # probability = xception_pred[0]
    # print("Xception Predictions:")
    # if probability[0] > 0.5:
    #     xception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
    # else:
    #     xception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
    # print(xception_chest_pred)

    # return render_template('results_chest.html',resnet_chest_pred=resnet_chest_pred,vgg_chest_pred=vgg_chest_pred,inception_chest_pred=inception_chest_pred,xception_chest_pred=xception_chest_pred)
    return_content = content
    return make_response(jsonify(return_content), 200)
app_dash = dash.Dash(server=app,url_base_pathname='/EE/')
# Generate the figure you need.
app_dash.layout = dcc.Graph(figure=my_figure(), style={"width": "100%", "height": "100vh"})
if __name__ == '__main__':
    app_dash.run_server()
