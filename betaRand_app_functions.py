import pandas as pd
import numpy as np
import datetime as dt
import random
from itertools import cycle, repeat
import io
from dateutil import parser
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource, FactorRange, Span, NumeralTickFormatter, HoverTool
from bokeh.transform import factor_cmap


sns.set_style("whitegrid")

def standardize_columns(data):
    data.columns = map(str, data.columns) 
    data.columns = [''.join(str.lower(e) for e in string if e.isalnum()) for string in data.columns] # replace all special characters in columns.
    df_str_columns = data.select_dtypes(exclude=[np.datetime64,np.number])
    for cols in df_str_columns.columns:
        print(cols)
        print(df_str_columns[cols])
        data[cols] = df_str_columns[cols].str.strip()#data[cols].str.strip()
        try:
            data[cols] = data[cols].map(lambda x: str(x).replace('-', ''))
        except UnicodeEncodeError:
            data[cols] = data[cols].map(lambda x: x.replace('-', ''))
    return data

def group_age(df,strat_columns):
    numeric_df = df[strat_columns].select_dtypes(include=[np.number])
    age_copy = []
    age_index = []
    if not numeric_df.empty:
        print("non-empty numeric dataframe")
        for cols in numeric_df.columns:
            age_copy = age_copy.append(df[cols])
            df[cols] = pd.qcut(df[cols].astype('float'),4).astype('str')
            age_index = age_index.append(df.index)
        return df, age_copy, age_index
    else:
        return df, [], []

def stratify(data_set,strat_columns,pure_randomization_boolean,sample_p, pure_randomization_text='Pure randomization'):
    """ 
    Stratified random sampling
    SPECIAL CASE, WHEN THERE IS ONLY ONE STRATUM PER INDIVIDUAL.
    * The test_size = 1 should be greater or equal to the number of classes = 5
    * Keep this in mind: https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/model_selection/_split.py#L1190
    """
    print("Stratify function")
    print("data_set")
    #print( data_set)
    data_set.dropna(axis=1,inplace=True)#,how='all')
    data_set = data_set.apply(lambda x: x.astype(str).str.lower())
    n = np.ceil((sample_p/100.)*len(data_set))

    print("data set WHY")
    #print(data_set)

    numeric_strat_columns = ~data_set[strat_columns].select_dtypes(include=[np.number]).empty
    if numeric_strat_columns:
        data_set, age_copy, age_index = group_age(data_set,strat_columns)

    if not pure_randomization_boolean:
        # - size of each group
        df = data_set.groupby(strat_columns).count().max(axis=1)
        #df = data_set.groupby(strat_columns).size() # Would this work?
        df = df.reset_index() # Create exception here

        # - How to ensure sample size when rounding like this.
        df['Size'] = np.ceil(n*(df[df.columns[-1]]/len(data_set)).values)

        # - Ensure that rounding of subgroups does not mess up total balance
        rows_delete = list(range(0,len(df)))
        random.shuffle(rows_delete)

        for rows in cycle(rows_delete):
            if df['Size'].sum() <= n:
                break
            else:
                df.loc[rows,'Size'] -= 1
        # And then cut from the larger groups.
        i=0
        ind_list=np.array([])

        for index,comb in df.iterrows():
            df_tmp = data_set[(data_set[comb[:-2].index]==comb[:-2].values).all(axis=1)]
            ind_list = np.append(ind_list,df_tmp.sample(n=int(df['Size'].iloc[i])).index.values)
            i += 1
    else:
        ind_list = data_set.sample(n=int(n)).index.values

    data_set['group-rct'] = ["intervention" if x in ind_list else "control" for x in data_set.index]

    todaysdate = str(dt.datetime.today().date())
    
    data_set['date'] = todaysdate
    data_set['date'] = pd.to_datetime(data_set['date']).dt.date
    data_set['batch'] = int(1)
    #total_data['date'] = total_data['date'].dt.strftime('%M/%d/%Y')

    if numeric_strat_columns:
        print("age_copy")
        print(age_copy)
        for or_,col_ in enumerate(data_set[strat_columns].select_dtypes(include=[np.number]).columns):
            data_set.loc[age_index[or_],col_] = age_copy[or_] 
    
    filename = 'test'

    if not pure_randomization_boolean: 
        name = filename.rsplit("--")[0]+"--"+",".join(strat_columns)+'_'+todaysdate+'_'+str(int(len(data_set)))+'_'+str(int(int(100-sample_p)))+'_RCT'+'.xlsx'
    else:
        name = filename.rsplit("--")[0]+"--"+str(pure_randomization_text)+'_'+todaysdate+'_'+str(int(len(data_set)))+'_'+str(int(int(100-sample_p)))+'_RCT'+'.xlsx'
     
    data_set.to_excel(name, na_rep='',index=False)

    return data_set, name

def update_stratification(data_set, data_new, filename1, pure_randomization_boolean, strat_columns, pure_randomization_text = 'Pure randomization'):
    """ 
    Stratified random sampling
    SPECIAL CASE, WHEN THERE IS ONLY ONE STRATUM PER INDIVIDUAL.
    RAISE MANY ERRORS.
    * The test_size = 1 should be greater or equal to the number of classes = 5
    * 
    * Keep this in mind: https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/model_selection/_split.py#L1190
    """

    p = .5#int(sample_p)/100. 

    sample_p = 50.

    data_set.index = data_set.index.astype('int')
    data_new.index = data_new.index.astype('int')

    data_set.dropna(axis=0,inplace=True,how='all',subset=data_set.columns[2:])
    data_set.dropna(axis=1,inplace=True,how='all')
    try:
        data_set = data_set.apply(lambda x: x.astype(str).str.lower())
    except UnicodeEncodeError:
        pass

    data_new.dropna(axis=0,inplace=True,how='all',subset=data_new.columns[2:])
    data_new.dropna(axis=1,inplace=True,how='all')
    try:
        data_new = data_new.apply(lambda x: x.astype(str).str.lower())
    except UnicodeEncodeError:
        pass

    data_new.columns = [x.lower() for x in data_new.columns]
    data_new.columns = data_new.columns.str.replace('\s+', '')

    data_set['date'] = data_set['date'].apply(lambda x: parser.parse(x.split('t')[0]))
    #data_set['date'] =  pd.to_datetime(data_set['date'], format='%Y%b%d')

    data_set_copy = data_set
    todaysdate = str(dt.datetime.today().date())
    data_new['date'] = todaysdate

    data_new['group-rct'] = ''
    data_temp = data_new.append(data_set.ix[:, :]) # there will be a problem with indexing, I can see it coming.

    if pure_randomization_boolean and pure_randomization_text in strat_columns:
        strat_columns.remove(pure_randomization_text)

    numeric_strat_columns = ~data_set[strat_columns].select_dtypes(include=[np.number]).empty
    if numeric_strat_columns:
        data_temp, age_copy, age_index = group_age(data_temp, strat_columns)

    #data_set = data_temp[data_temp.date != todaysdate] # seleccionar datos ya asignados
    data_set = data_temp[(data_temp['group-rct'].isin(['control','intervention']))] # seleccionar datos ya asignados

    label = str(((data_set_copy['group-rct'].value_counts(normalize=True)-p)).idxmin()) # los que se quedan bajitos
    initial_n = data_set_copy['group-rct'].value_counts().loc[label] # size de los que se quedan bajitos

    if not pure_randomization_boolean:
        df = data_temp.groupby(strat_columns).size().reset_index() # Number of individuals in each group

        label_pre = pd.crosstab(data_set['group-rct'],[pd.Series(data_set[cols]) for cols in strat_columns]).loc[label].reset_index() 

        # desired size
        if label == 'control':
            n = np.ceil(p*len(data_temp))
        elif label == 'intervention':
            n = np.ceil((1-p)*len(data_temp)) 
        else:
            print("ERROOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOR")
        df['Size'] = np.ceil(n*(df[df.columns[-1]]/len(data_temp)).values) # number of individuals in the selected intervention that would make up for a balanced contribution to the covariates


        rows_delete = list(range(0,len(df)))
        random.shuffle(rows_delete)

        previous_assignation = df['Size'].sum()

        # -- This cicle determines the sizes of the groups given by the covariates
        #print(n-initial_n)
        #print(df['Size'].sum())
        if (n) < df['Size'].sum(): #- initial_n
            #print("Sobran individuos por aleatorizar")
            deleted_ns = 0  
            for rows in cycle(rows_delete):
                #print("Entramos al ciclo")
                if df.loc[rows,'Size'] > 0:
                    #((1/p)*df['Size'].sum())-n:
                    #print(deleted_ns)
                    if deleted_ns >= (previous_assignation - n):#- (n - initial_n)): #n - initial_n - df['Size'].sum() :
                        print("Rompimos")
                        break
                    else:
                        df.loc[rows,'Size'] -= 1
                        deleted_ns += 1
                        print("lo que falta "+str(n - df['Size'].sum())) #- initial_n 
                        #print(deleted_ns)   
                        #print(df)
        elif (n) > df['Size'].sum(): #- initial_n
            added_ns = 0 
            for rows in cycle(rows_delete):
                if df.loc[rows,'Size'] > 0:
                    if added_ns >= ((n - initial_n) - previous_assignation ):
                        break
                    else:
                        df.loc[rows,'Size'] += 1
                        added_ns += 1
        
        df = df.merge(label_pre)
        df['Missing'] = df['Size'] - df[label] # difference between existing and needed amounts
        ind_list = np.array([]) #  Maybe shuffle data_new a little bit
        diff = n - (data_set_copy['group-rct']==label).sum() # desired number of individuals to fill out in the 'label' group. Same as n - initial_n, I hope.
        assigned = 0

        #data_new = data_temp[data_temp.date==todaysdate] # what happens if an update happens on the same day?
        #data_ = data_temp[data_temp['group-rct'].isnull()]
        data_new = data_temp[~(data_temp['group-rct'].isin(['control','intervention']))]

        # -- This cycle assigns groups at random given our corrected sizes.
        for index,comb in df.iterrows():
            # This should ensure that we are not filling in more numbers than necessary (but not the inverse)
            if assigned >= n - initial_n:
                break
            else:
                df_tmp = data_new[(data_new[comb[:-4].index]==comb[:-4].values).all(axis=1)]  # Combinations of factors.
                sz = len(df_tmp)
                ss = min([sz, df['Missing'].loc[index], diff]) # What I have vs. what I am missing, only god knows why diff is here.
                if ss > 0:
                    #print(ss)
                    ind_list = np.append(ind_list, df_tmp.sample(n=int(ss)).index.values)
                    assigned += ss
                else:
                    pass
        # Arreglando esa colita
        if assigned < diff:
            elegible = data_temp[(~(data_temp.index.isin(ind_list)))&(data_temp['group-rct']=='')&(data_temp['date']==todaysdate)]
            available = min(int(diff)-assigned,len(data_temp[data_temp['date']==todaysdate]))
            if len(elegible) >= available:
                ind_list_b = elegible.sample(int(available)).index.values
                ind_list   = np.append(ind_list,ind_list_b)
            else:
                ind_list_b = elegible.index.values
                ind_list   = np.append(ind_list,ind_list_b)
        ind_list = map(int, ind_list)
    else: 
        label_pre = data_set['group-rct'].value_counts().loc[label] 

        if label == 'control':
            n = np.ceil(p*len(data_temp))
        elif label == 'intervention':
            n = np.ceil((1-p)*len(data_temp)) 
        else:
            print("ERROOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOR")
     
        ind_list = data_new.sample(int(n)-int(data_set['group-rct'].value_counts().loc[label])).index.values

    print("data_new")
    #print(data_new)
    print("ind_list")
    #print(ind_list)
    if label == 'control':
        data_new.loc[ind_list,'group-rct'] = "control"
        data_new.loc[set(data_new.index.values ) - set(ind_list),'group-rct'] = "intervention"
        for x in ind_list:
            if (x not in data_new.index):
                print("malo")
                print(x)
    else:
        #print("data_new.index")
        #print(data_new.index)
        #print("data_new.loc[ind_list]")
        #print(data_new.loc[ind_list])
        data_new.loc[ind_list,'group-rct'] = "intervention"
        data_new.loc[set(data_new.index.values ) - set(ind_list),'group-rct'] = "control"


    todaysdate = str(dt.datetime.today().date())
    data_new['batch'] = int(np.max(data_set.batch.value_counts().index.astype('int').values)) + int(1)

    total_data = data_new.append(data_set)
    if numeric_strat_columns:
        print("age_copy")
        print(age_copy)
        for or_,col_ in enumerate(data_set[strat_columns].select_dtypes(include=[np.number]).columns):
            data_set.loc[age_index[or_],col_] = age_copy[or_]

    if not pure_randomization_boolean: 
        name = filename1.rsplit("--")[0]+"--"+",".join(strat_columns)+'_'+todaysdate+'_'+str(int(len(total_data)))+'_'+str(int(sample_p))+'_RCT'+'.xlsx'
        name_static = filename1.rsplit("--")[0]+"--"+",".join(strat_columns)+'_'+todaysdate+'_'+str(int(len(total_data)))+'_'+str(int(sample_p))+'.xlsx'
    else:
        name = filename1.rsplit("--")[0]+"--"+str(pure_randomization_text)+'_'+todaysdate+'_'+str(int(len(total_data)))+'_'+str(int(sample_p))+'_RCT'+'.xlsx'
        name_static = filename1.rsplit("--")[0]+"--"+str(pure_randomization_text)+'.xlsx'+'_'+todaysdate+'_'+str(int(len(total_data)))+'_'+str(int(sample_p))+'.xlsx'
 

    total_data = total_data.set_index(data_set.columns[0])

    data_new.to_excel(name_static, na_rep='',index=False)
    total_data.to_excel(name, na_rep='')

    name_log = filename1.rsplit(".")[0]+todaysdate+'_log.xlsx'
    writer = pd.ExcelWriter(name_log, engine = 'xlsxwriter')
    for col in strat_columns:
        if col=="age":
            pass
        else:
            pd.crosstab(total_data[col], total_data['group-rct']).to_excel(writer, sheet_name=col)
    writer.save()

    return data_set, strat_columns, name

def check_strat_file(data_rct, data_new, filename1, pure_randomization_text = 'Pure randomization'):

    valid_update = False
    pure_randomization_boolean = False
    strat_columns = []
    message_update = ''
    data_rct.dropna(axis=1,how='all',inplace=True)
    data_rct.dropna(axis=0,how='all',inplace=True)

    data_rct.columns = map(str, data_rct.columns)
    available_columns = []
    try:
        available_columns = list(set(data_rct.columns.values) - set(['group-rct','date','batch']))
    except:
        pass

    data_rct = data_rct.rename(columns={string:''.join(str.lower(e) for e in string if e.isalnum()) for string in available_columns}) #remove all special characters
    try:
        data_rct = data_rct.apply(lambda x: x.astype(str).str.lower())
        data_rct = standardize_columns(data_rct)
    except UnicodeEncodeError:
        data_rct.replace(r'[,\"\']','', regex=True).replace(r'\s*([^\s]+)\s*', r'\1', regex=True, inplace=True)

    data_new.dropna(axis=1,how='all',inplace=True)
    data_new.dropna(axis=0,how='all',inplace=True,subset=data_new.columns[2:])
    data_new = standardize_columns(data_new)
    data_new = data_new.apply(lambda x: x.astype(str).str.lower())
    
    data_new.columns = [''.join(str.lower(str(e)) for e in string if e.isalnum()) for string in data_new.columns]
    data_new.replace(r'[,\"\']','', regex=True).replace(r'\s*([^\s]+)\s*', r'\1', regex=True, inplace=True)
    
    data_new.columns = map(str, data_new.columns)
    data_new.columns = map(str.lower, data_new.columns)
    data_new.columns = data_new.columns.str.replace(' ','')

    sample_p = float(filename1.rsplit("_")[-2])

    if 'grouprct' in data_rct.columns:
        if set(data_rct.columns)-set(['grouprct','date','batch']) == set(data_new.columns):
            #CHANGE THIS
            try:
                if len(filename1.rsplit("--")) <=1:
                    message_update = "Please check the naming structure of the mother file."
                else:
                    valid_update = True
                    strat_columns = filename1.rsplit("--")[-1].rsplit("_")[0].rsplit(",")
                    if strat_columns[0] != pure_randomization_text:
                        strat_columns = [''.join(c.lower() for c in x if not c.isspace()) for x in strat_columns]
                    else:
                        pure_randomization_boolean = True
            except IndexError:
                strat_columns = []
                pass

            common_columns = list(set(data_rct.columns[1:]) & set(data_new.columns[1:]))
            new_categories = {}
            if not pure_randomization_boolean:
                for col in strat_columns:
                    if col!='age':
                        new_values = set(data_new[col]) - set(data_rct[col])
                        if new_values:
                            new_categories[col] = new_values
                if bool(new_categories):
                    # Need to treat this as a special category.
                    message_update = '''There are new values {1} in {0}.'''.format(new_categories.keys(),new_categories.values())       
                    # OJO
        else:
            available_columns = list(set(data_rct.columns.values) - set(['grouprct','date','batch']))
            message_update = "Files must have the same structure (columns). \n Previous column names: "+ str([x.encode('utf-8') for x in data_rct[available_columns].columns.values]) +"\n New column names: "+str([x.encode('utf-8') for x in data_new.columns.values])

    else:    
        message_update = "File should have been generated by this own program and thus have a Group-RCT column."          
        #"The file should have been generated by this own program and thus have a Group-RCT column."
    #pass
    return valid_update, message_update, pure_randomization_boolean, strat_columns, sample_p

def warning_new_words(new_categories):
    pass
    # Implement this warning, somehow.

def create_plots(data_rand, strat_columns, pure_randomization_boolean, sample_p,session_update):
    viz_list = []
    sample_p = sample_p / 100

    for i in range(len(strat_columns)):
        print(strat_columns[i])
        print(data_rand[strat_columns[i]])
        print(data_rand[strat_columns[i]].dtype)
        try:
            data_rand[strat_columns[i]] = pd.to_numeric(data_rand[strat_columns[i]])        
            print("SOMETHING NUMERIC")
            # fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
            # counts = [5, 3, 4, 2, 4, 6]
            # p = figure(x_range=fruits, 
            #             plot_height=250, 
            #             title="Fruit Counts",
            #             toolbar_location=None, 
            #             tools="")
            # p.vbar(x=fruits, top=counts, width=0.9)number):

            df = data_rand.groupby('group-rct').agg(['mean','std'])[strat_columns[i]]
            print(df)
            #(100*(pd.crosstab(data_rand['group-rct'], data_rand[strat_columns[i]], normalize='columns')))
            #df = df.stack().reset_index().rename(columns={0:'Percentage'}) 
            colors = ["#c9d9d3", "#718dbf"]
            p = figure(x_range=list(pd.unique(df.index)), 
                        plot_height=250, 
                        title=strat_columns[i],
                        toolbar_location=None, 
                        tools="")
            p.vbar(x=list(df.index.values), 
                top=list(df['mean'].values), 
                width=0.9, color=["#c9d9d3", "#718dbf"])

        except ValueError:
            print("NOT NUMERIC")
            #print(data_rand)
            df = pd.crosstab(data_rand['group-rct'], data_rand[strat_columns[i]], normalize='columns')#.set_index([])
            # df = df.stack().reset_index().rename(columns={0:'Percentage'}) 
            print(df)
            palette = ["#c9d9d3", "#718dbf", "#e84d60"]
            # p = figure(x_range=list(df.index.values), 
            #             plot_height=250, 
            #             title=strat_columns[i],
            #             toolbar_location=None, 
            #             tools="")
            # p.vbar(x=list(df.index.values), 
            #     top=list(df['Percentage'].values), 
            #     width=0.9)
            x = [ (col, group) for col in df.columns for group in df.index.values]
            print(x)
            pcts_ = list(df.transpose().stack().values) # like an hstack
            print(pcts_)
            
            source = ColumnDataSource(data=dict(
                x=x, counts=pcts_, sample_p=list(repeat(sample_p, len(x)))))

            hover = HoverTool(
                        tooltips = [
                            ('Percentage', '@counts{%0.2f}'),
                            ('Goal for control group', '@sample_p{%0.2f}')
                        ],
                        formatters = {'Percentage': 'numeral', 'Goal for control group':'numeral'}
                    )

            p = figure(x_range=FactorRange(*x), plot_height=350, title=strat_columns[i],
                       toolbar_location=None, tools=[hover])
            
            p.yaxis.formatter = NumeralTickFormatter(format='0 %')

            p.vbar(x='x', top='counts', width=0.9, source=source, line_color="white",
                fill_color=factor_cmap('x', palette=palette, factors=list(df.index.values), start=1, end=2))

            sample_p_ref = Span(location=sample_p, dimension='width', 
                line_color='red', line_dash='dashed', line_width=3)

            p.add_layout(sample_p_ref)

        script, div = components(p)
        viz_list.append((script,div))
    return viz_list


