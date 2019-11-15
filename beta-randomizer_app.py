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

HTML_HEAD = '<title>Beta-Randomizer</title> \
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous"> \
<script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script> \
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script> \
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script> '

UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = set(['xlsx','xlx'])

app = Flask(__name__)
app.secret_key = "password"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def choose_rand_option():
    session.clear()
    session['data'] = pd.DataFrame([]).to_json()
    if request.method == 'POST':# NEED TO TURN THESE INTO POST REQUESTS INSTEAD OF REDIRECTS.
        if request.form['desired_action'] == 'create':
            return redirect(url_for('create_scheme'))
        if request.form['desired_action'] == 'update':
            return redirect(url_for('update_scheme'))
    if request.method == 'GET':
        return render_template('welcome.html', HTML_HEAD=HTML_HEAD)

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
            #print('No file part')
            return redirect(request.url)
        file_new = request.files['file_new']
        file_RCT= request.files['file_RCT']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file_new.filename == '' or file_RCT.filename == '':
            flash('No selected file')
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
        #session['data'] = False
        session['update'] = False

        return render_template('update_scheme.html')

@app.route('/visualize_scheme', methods=['GET','POST'])
def visualize_scheme():
    if request.method == 'POST':
        #print("session['update']")
        #print(session['update'])
        data_rand = pd.DataFrame([])
        strat_columns = []
        pure_randomization_boolean = False
        sample_p = 50.
        if session['update']==True:
            session_update=True
            data_rct = pd.DataFrame(ast.literal_eval(session['data_rct']))
            data_new = pd.DataFrame(ast.literal_eval(session['data_new']))
            filename1 = session['filename1']
            valid_update, message_update, pure_randomization_boolean, strat_columns, sample_p = check_strat_file(data_rct, data_new, session['filename1'])
            if valid_update:
                data_rand, strat_columns = update_stratification(data_rct, data_new, session['filename1'], pure_randomization_boolean, strat_columns)
                session['data_rand'] = data_rand.to_json()
            else:
                flash(message_update)
                #redirect(url_for('update_scheme'))
        else:
            session_update=False
            # THERE WILL BE AN ERROR HERE. GET with update vs get without an update.
            print("create statement")
            data_set = pd.DataFrame(ast.literal_eval(session['data']))
            strat_columns = []
            sample_p = int(request.form.get('sample_p'))#int(request.form['sample_p'])
            for cols in data_set.columns:
                if request.form.get(cols) == '1':
                    strat_columns.append(str(cols))

            rand_type = request.form.get('randomization_type')
            if rand_type == 'Simple':
                data_rand = stratify(data_set, strat_columns,
                                        pure_randomization_boolean=True, 
                                        sample_p=sample_p)

            elif rand_type == 'Stratified':
                data_rand = stratify(data_set, strat_columns,
                                    pure_randomization_boolean=False, 
                                    sample_p=sample_p)

        if (not data_rand.empty):
            data_rand.to_excel("data_rand.xlsx")
            viz_list = create_plots(data_rand,strat_columns,pure_randomization_boolean,sample_p,session_update)
            
            #return render_template('visualize_scheme.html', data_rand = data_rand, plot_url = plot_url)#send_file(app.config['UPLOAD_FOLDER']+"/the-global-city-brown.pdf", as_attachment=True)
            return render_template('visualize_scheme.html', viz_list=viz_list)

                    #send_from_directory(app.config['UPLOAD_FOLDER'], "the-global-city-brown.pdf", as_attachment=True), \
    else:
        return '''This should never happen.'''
    
@app.route('/df_download/<filename>',methods=['GET','POST'])
def df_download(filename):
    return send_file(app.config['UPLOAD_FOLDER']+"/data_rand.xlsx", as_attachment=True)
    #return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    app.debug = True
    #app.run(host='0.0.0.0', port=4500)
    app.run()
