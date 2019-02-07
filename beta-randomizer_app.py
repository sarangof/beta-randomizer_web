import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, session, send_file
from werkzeug.utils import secure_filename
from betaRand_app_functions import stratify, standardize_columns, check_strat_file, update_stratification, create_plots
import pandas as pd
import ast

import base64
import matplotlib.pyplot as plt
#global data 



from dateutil import parser

data = pd.DataFrame([])

UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = set(['csv','xlsx','xlx'])

app = Flask(__name__)
app.secret_key = "password"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def choose_rand_option():
    session['data'] = pd.DataFrame([]).to_json()
    if request.method == 'POST':# NEED TO TURN THESE INTO POST REQUESTS INSTEAD OF REDIRECTS.
        if request.form['desired_action'] == 'create':
            return redirect(url_for('create_scheme'))
        if request.form['desired_action'] == 'update':
            return redirect(url_for('update_scheme'))
    if request.method == 'GET':
        return render_template('welcome.html')  

@app.route('/create_scheme', methods=['GET', 'POST'])
def create_scheme():
    if request.method == 'POST':
        session['update'] = False
        print("I was here at least")
        if 'file' not in request.files:
            flash('No file part')
            print('No file part')
            return redirect(request.url)

        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            print('No selected file')
            return redirect(request.url)#redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_all = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_all)
            
            session['data'] = pd.read_excel(file_all).to_json()
            print("session['data']")
            print(session['data'])
            return redirect(url_for('randomization_options'), code=307)#redirect(url_for('randomization_options'))

    else:
        #session['data'] = None
        return render_template('create_scheme.html')#redirect(request.url)#render_template('create_scheme.html',code=302)


@app.route('/randomization_options',methods=['GET','POST'])
def randomization_options():
    if request.method == 'POST':
        print("session['data']")
        print(session['data'])
        if session['data'] is not None:
            #print(session['data'])
            data = standardize_columns(pd.DataFrame(ast.literal_eval(session['data'])))
            session['data'] = data.to_json()
            return render_template('new-scheme-options.html', columns=data.columns)
        else:
            return '''What should I do here?'''

@app.route('/update_scheme', methods=['GET','POST'])
def update_scheme():
    if request.method == 'POST':
        print("postUPDATESCHEME")
        session['update'] = True
        if (('file_new' not in request.files) or ('file_RCT' not in request.files)):
            flash('No file part')
            print('No file part')
            return redirect(request.url)
        file_new = request.files['file_new']
        file_RCT= request.files['file_RCT']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file_new.filename == '' or file_RCT.filename == '':
            flash('No selected file')
            print('No selected file')
            return redirect(request.url)

        if file_new and file_RCT and allowed_file(file_new.filename) and allowed_file(file_RCT.filename):

            print("allowed files for scheme updation")
            filename_RCT = secure_filename(file_RCT.filename)
            filename_new = secure_filename(file_new.filename)

            file_all_RCT = os.path.join(app.config['UPLOAD_FOLDER'], filename_RCT)
            file_all_new = os.path.join(app.config['UPLOAD_FOLDER'], filename_new)

            file_RCT.save(file_all_RCT)
            file_new.save(file_all_new)
            
            session['data_rct'] = pd.read_excel(file_all_RCT).to_json(date_format='iso', date_unit='s')
            #print("session['data_rct'] JSON VERSION")
            #print(session['data_rct'])
            session['data_new'] = pd.read_excel(file_all_new).to_json()
            session['filename1'] = str(file_RCT.filename)

            return redirect(url_for('visualize_scheme'),code=307)

    else:
        print("GETUPDATESCHEME")
        print("session['update']")
        print(session['update'])
        #session['data'] = False
        #session['update'] = False

        return render_template('update_scheme.html')

@app.route('/visualize_scheme', methods=['GET','POST'])
def visualize_scheme():
    if request.method == 'POST':
        #print("session['update']")
        #print(session['update'])
        data_rand = pd.DataFrame([])
        strat_cols = []
        print("session['update']")
        print(session['update'])
        if session['update']==True:
            sample_p = 50.
            data_rct = pd.DataFrame(ast.literal_eval(session['data_rct']))
            print("data_rct['date']")
            print(data_rct['date'])
            data_new = pd.DataFrame(ast.literal_eval(session['data_new']))
            filename1 = session['filename1']
            valid_update, message_update, pure_randomization_boolean, strat_columns = check_strat_file(data_rct, data_new, session['filename1'])
            print("valid_update")
            print(valid_update)
            print("message_update")
            print(message_update)
            print("check validity")
            print(valid_update==True)
            if valid_update:
                data_rand, strat_cols = update_stratification(data_rct, data_new, session['filename1'], pure_randomization_boolean, strat_columns)
                print("data_rand")
                print(data_rand)
                print("strat_cols")
                print(strat_cols)
                session['data_rand'] = data_rand.to_json()  
                print("valid update")
            else:
                flash(message_update)
                #redirect(url_for('update_scheme'))
        else:
            # THERE WILL BE AN ERROR HERE. GET with update vs get without an update.
            print("create statement")
            sample_p = 50.
            data_set = pd.DataFrame(ast.literal_eval(session['data']))
            strat_cols = []
            for cols in data_set.columns:
                if request.form.get(cols) == '1':
                    print(cols)
                    strat_cols.append(str(cols))
                    print(strat_cols)
            data_rand = stratify(data_set,strat_cols)
            print("stratifying succesfull")
            print(data_rand)

        if (not data_rand.empty):
            print("not data-rand empty")
            data_rand.to_excel("data_rand.xlsx")
            print("saved to excel")
            #img = create_plots(data_rand,strat_cols)      
            #print("plots created")
            #plt.savefig(img, format='png')
            #print("image created")
            #img.seek(0)
            #plot_url = base64.b64encode(img.getvalue()).decode()
            #return '<img src="data:image/png;base64,{}">'.format(plot_url)
            #return render_template('visualize_scheme.html', data_rand = data_rand)
            print("data_rand")
            print(data_rand)
            
            #return render_template('visualize_scheme.html', data_rand = data_rand, plot_url = plot_url)#send_file(app.config['UPLOAD_FOLDER']+"/the-global-city-brown.pdf", as_attachment=True)
            return render_template('visualize_scheme.html', data_rand = data_rand, plot_url = [])

                    #send_from_directory(app.config['UPLOAD_FOLDER'], "the-global-city-brown.pdf", as_attachment=True), \
    else:
        return '''This should never happen.'''
    
@app.route('/df_download/<filename>',methods=['GET','POST'])
def df_download(filename):
    return send_file(app.config['UPLOAD_FOLDER']+"/data_rand.xlsx", as_attachment=True)
    #return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=4500)
