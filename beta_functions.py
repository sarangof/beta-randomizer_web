
#!/usr/bin/python2.7
import pandas as pd
import numpy as np
import datetime as dt
import random
from itertools import cycle

def stratify(self):
    """ 
    Stratified random sampling
    SPECIAL CASE, WHEN THERE IS ONLY ONE STRATUM PER INDIVIDUAL.
    * The test_size = 1 should be greater or equal to the number of classes = 5
    * Keep this in mind: https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/model_selection/_split.py#L1190
    """

    #data = pd.read_csv('Randomization_short.csv') #'NYU - Franklin JIYA to randomizecaseloads.xlsx'
    data_set = self.data
    selected_columns = self.strat_columns
    data_set.dropna(axis=1,inplace=True)#,how='all')
    data_set = data_set.apply(lambda x: x.astype(str).str.lower())
    n = np.ceil((self.sample_p/100.)*len(data_set))
    #print(selected_columns)
    #print(data_set.head())

    if "age" in self.strat_columns:
        data_set, age_copy, age_index = group_age(data_set)

    if not self.pure_randomization_text:
        # - size of each group
        df = data_set.groupby(selected_columns).count().max(axis=1)
        #df = data_set.groupby(selected_columns).size() # Would this work?
        df = df.reset_index() # Create exception here

        #print(n)
        #print("df")
        #print(df)

        # - How to ensure sample size when rounding like this.
        df['Size'] = np.ceil(n*(df[df.columns[-1]]/len(data_set)).values)

        # - Ensure that rounding of subgroups does not mess up total balance
        rows_delete = range(0,len(df))
        random.shuffle(rows_delete)

        for rows in cycle(rows_delete):
            if df['Size'].sum() <= n:
                break
            else:
                df.loc[rows,'Size'] -= 1

        #print("df size")
        #print(df)

        # And then cut from the larger groups.
        i=0
        ind_list=np.array([])
        for index,comb in df.iterrows():
            df_tmp = data_set[(data_set[comb[:-2].index]==comb[:-2].values).all(axis=1)]
            ind_list = np.append(ind_list,df_tmp.sample(n=df['Size'].iloc[i]).index.values)
            i += 1
    else:
        #print("index")
        #print(n)
        ind_list = data_set.sample(n=int(n)).index.values

    data_set['group-rct'] = ["intervention" if x in ind_list else "control" for x in data_set.index]

    todaysdate = str(dt.datetime.today().date())
    
    data_set['date'] = todaysdate
    data_set['date'] = pd.to_datetime(data_set['date']).dt.date
    data_set['batch'] = int(1)
    #self.total_data['date'] = self.total_data['date'].dt.strftime('%M/%d/%Y')

    if "age" in self.strat_columns:
        data_set.loc[age_index,'age'] = age_copy 
    
    if not self.pure_randomization_boolean:
        name_log = self.filename.rsplit("|")[0]+"|"+",".join(selected_columns)+'-'+str(int(100-self.sample_p))+str(dt.datetime.now())+'_log.xlsx'
        self.name = self.filename.rsplit(".")[0]+"|"+",".join(selected_columns)+'_'+str(todaysdate)+'_'+str(int(len(data_set)))+'_'+str(int(100-self.sample_p))+'_RCT'+'.xlsx'
        writer = pd.ExcelWriter(name_log, engine = 'xlsxwriter')
        for col in self.strat_columns:
            if col == "age":
                pass
            else:
                pd.crosstab(data_set[col], data_set['group-rct']).to_excel(writer, sheet_name=col)
    else:
        name_log = self.filename.rsplit("|")[0]+"|"+str(self.pure_randomization_text)+'-'+str(int(100-self.sample_p))+str(dt.datetime.now())+'_log.xlsx'
        self.name = self.filename.rsplit(".")[0]+"|"+str(self.pure_randomization_text)+'_'+str(todaysdate)+'_'+str(int(len(data_set)))+'_'+str(int(100-self.sample_p))+'_RCT'+'.xlsx'
        writer = pd.ExcelWriter(name_log, engine = 'xlsxwriter')
        data_set['group-rct'].value_counts().to_excel(writer, sheet_name=self.pure_randomization_text)
    writer.save()

    data_set.to_excel(self.name, na_rep='',index=False)

    return self.name

def update_stratification(self):
    """ 
    Stratified random sampling
    SPECIAL CASE, WHEN THERE IS ONLY ONE STRATUM PER INDIVIDUAL.
    RAISE MANY ERRORS.
    * The test_size = 1 should be greater or equal to the number of classes = 5
    * 
    * Keep this in mind: https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/model_selection/_split.py#L1190
    """

    #data = pd.read_csv('Randomization_short.csv') #'NYU - Franklin JIYA to randomizecaseloads.xlsx'

    #data_set = pd.read_excel('Randomized/Originals Aug 22/NYU - Hamilton JIYA June 20,          AGE,RISK SCORE,Gender,Race-50_RCT.xlsx')
    #data_new = pd.read_excel('To randomize Aug 22/Hamilton County cases to be randomized 8-15-17.xlsx')

    p = int(self.sample_p)/100. #percentual proportion
    #print("p")
    #print(self.sample_p)
    #print("Existing data")
    #print(self.data_rct.head())

    #print("New data")
    #print(self.data_new.head())

    data_set = self.data_rct#pd.read_excel(filename)
    data_set.dropna(axis=0,inplace=True,how='all',subset=data_set.columns[2:])
    data_set.dropna(axis=1,inplace=True,how='all')
    try:
        data_set = data_set.apply(lambda x: x.astype(str).str.lower())
    except UnicodeEncodeError:
        pass

    data_new = self.data_new


    #data_set=pd.read_excel('ICAN Texting Trial_Randomization List_6-27-18_V2|risklevel,age,sex,race,mentoring,sentencedordetained_2018-07-18_31_50_RCT.xlsx')
    #data_new=pd.read_excel('ICAN Texting Trial_Randomization List_7-5-18.xlsx')


    data_new.dropna(axis=0,inplace=True,how='all',subset=data_new.columns[2:])
    data_new.dropna(axis=1,inplace=True,how='all')
    try:
        data_new = data_new.apply(lambda x: x.astype(str).str.lower())
    except UnicodeEncodeError:
        pass

    data_new.columns = [x.lower() for x in data_new.columns]
    data_new.columns = data_new.columns.str.replace('\s+', '')

    data_set_copy = data_set
    todaysdate = str(dt.datetime.today().date())
    data_new['date'] = todaysdate

    data_new['group-rct'] = ''
    data_temp = data_new.append(data_set.ix[:, :]) # there will be a problem with indexing, I can see it coming.
    data_temp, age_copy, age_index = group_age(data_temp)

    #print(selected_columns)

    #print("RCT data:")
    #print(data_set.head())

    #print("New data:")
    #print(data_new.head())

    #data_set = data_temp[data_temp.date != todaysdate] # seleccionar datos ya asignados
    data_set = data_temp[(data_temp['group-rct'].isin(['control','intervention']))] # seleccionar datos ya asignados
    print(data_set)
    #df = data_set.groupby(selected_columns).size().reset_index() # Number of individuals in each group

    label = str(((data_set_copy['group-rct'].value_counts(normalize=True)-p)).idxmin()) # los que se quedan bajitos
    initial_n = data_set_copy['group-rct'].value_counts().loc[label] # size de los que se quedan bajitos

    print("pure randomization boolean")
    print(self.pure_randomization_boolean)

    if not self.pure_randomization_boolean:
        selected_columns = self.strat_columns 
        print("columns")
        print(selected_columns)
        df = data_temp.groupby(selected_columns).size().reset_index() # Number of individuals in each group

        #print("label")
        #print(label)

        #print("CROSSTAB")
        #print(pd.crosstab(data_set['group-rct'],[pd.Series(data_set[cols]) for cols in selected_columns]))

        label_pre = pd.crosstab(data_set['group-rct'],[pd.Series(data_set[cols]) for cols in selected_columns]).loc[label].reset_index() 

        # desired size
        if label == 'control':
            n = np.ceil(p*len(data_temp))
        elif label == 'intervention':
            n = np.ceil((1-p)*len(data_temp)) 
        else:
            print("ERROOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOR")
        df['Size'] = np.ceil(n*(df[df.columns[-1]]/len(data_temp)).values) # number of individuals in the selected intervention that would make up for a balanced contribution to the covariates
        # Trying to get a more exact sample size.
        #print("ESTE VALOR")        
        #print(n-df['Size'].sum())

        print('Grupos share')
        print(data_set_copy['group-rct'].value_counts(normalize=True))
        print(data_set_copy['group-rct'].value_counts())

        print("n: "+str(n))
        print("p: "+str(p))
        print("initial_n: "+str(initial_n))
        print("label: "+str(label))
        print("Actual assignation: "+str(df['Size'].sum()))

        rows_delete = range(0,len(df))
        random.shuffle(rows_delete)

        #print("DF BEFORE")
        #print(df)

        previous_assignation = df['Size'].sum()

        # -- This cicle determines the sizes of the groups given by the covariates
        print(n-initial_n)
        print(df['Size'].sum())
        if (n) < df['Size'].sum(): #- initial_n
            #print("Sobran individuos por aleatorizar")
            deleted_ns = 0  
            for rows in cycle(rows_delete):
                #print("Entramos al ciclo")
                if df.loc[rows,'Size'] > 0:
                    #((1/p)*df['Size'].sum())-n:
                    print(deleted_ns)
                    if deleted_ns >= (previous_assignation - n):#- (n - initial_n)): #n - initial_n - df['Size'].sum() :
                        print("Rompimos")
                        break
                    else:
                        df.loc[rows,'Size'] -= 1
                        deleted_ns += 1
                        print("lo que falta "+str(n - df['Size'].sum())) #- initial_n 
                        print(deleted_ns)   
                        print(df)
        elif (n) > df['Size'].sum(): #- initial_n
            print("Faltan individuos por aleatorizar")
            print("BUUUUUUUUUUUU - I would be surprised if this were not impossible. ")
            added_ns = 0 
            for rows in cycle(rows_delete):
                if df.loc[rows,'Size'] > 0:
                    if added_ns >= ((n - initial_n) - previous_assignation ):
                        break
                    else:
                        df.loc[rows,'Size'] += 1
                        added_ns += 1
                        print("lo que falta "+str(n - initial_n - df['Size'].sum()))
                        print(added_ns)   
                        print(df)
        #print("DF AFTER")
        #print(df)

        print("Current assignation - after")
        print(df['Size'].sum())
        
        df = df.merge(label_pre)
        df['Missing'] = df['Size'] - df[label] # difference between existing and needed amounts
        ind_list = np.array([]) #  Maybe shuffle data_new a little bit
        diff = n - (data_set_copy['group-rct']==label).sum() # desired number of individuals to fill out in the 'label' group. Same as n - initial_n, I hope.
        assigned = 0
        print("Missing")
        print(df['Missing'])
        print("diff")
        print(diff)

        #data_new = data_temp[data_temp.date==todaysdate] # what happens if an update happens on the same day?
        #data_ = data_temp[data_temp['group-rct'].isnull()]
        data_new = data_temp[~(data_temp['group-rct'].isin(['control','intervention']))]

        # -- This cycle assigns groups at random given our corrected sizes.
        for index,comb in df.iterrows():
            # This should ensure that we are not filling in more numbers than necessary (but not the inverse)
            if assigned >= n - initial_n:
                break
            else:
                #print("comb")
                #print(comb)
                #print("subset")
                #print(comb[:-4])
                df_tmp = data_new[(data_new[comb[:-4].index]==comb[:-4].values).all(axis=1)]  # Combinations of factors.
                #print("df_tmp")
                #print(df_tmp)
                #print(df_tmp.index)
                sz = len(df_tmp)
                ss = min([sz, df['Missing'].loc[index], diff]) # What I have vs. what I am missing, only god knows why diff is here.
                #print("missing")
                #print(df['Missing'].loc[index])
                if ss > 0:
                    print(ss)
                    ind_list = np.append(ind_list, df_tmp.sample(n=int(ss)).index.values)
                    assigned += ss
                else:
                    pass
                #print("assigned")
                #print(assigned)
        print("Assigned")
        print(assigned)
        print("diff")
        print(diff)
        print("ind list pre")
        print(ind_list)
        print("len ind list pre")
        print(len(set(ind_list)))
        # Arreglando esa colita
        if assigned < diff:
            print("Assigned are less assigned than the desired people to assign")
            elegible = data_temp[(~(data_temp.index.isin(ind_list)))&(data_temp['group-rct']=='')&(data_temp['date']==todaysdate)]
            print(elegible)
            available = min(int(diff)-assigned,len(data_temp[data_temp['date']==todaysdate]))
            print(available)

            if len(elegible) >= available:
                ind_list_b = elegible.sample(int(available)).index.values
                ind_list   = np.append(ind_list,ind_list_b)
            else:
                ind_list_b = elegible.index.values
                ind_list   = np.append(ind_list,ind_list_b)

        print("ind_list")
        print(ind_list)
        print("length")
        print(len(ind_list))
        # I can just randomly delete indices from the lists
        #ind_list = list(pd.DataFrame(ind_list).sample(n))


        #print(pd.DataFrame(ind_list)[0].value_counts())

        ind_list = map(int, ind_list)
        
        #print("data_new")
        #print(data_new)

    else: 
        label_pre = data_set['group-rct'].value_counts().loc[label] 

        # desired size
        if label == 'control':
            n = np.ceil(p*len(data_temp))
            #n = np.ceil((1-p)*len(data_temp)) 
        elif label == 'intervention':
            n = np.ceil((1-p)*len(data_temp)) 
            #n = np.ceil(p*len(data_temp))
        else:
            print("ERROOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOR")
        #df['Size'] = np.ceil(n*(df[df.columns[-1]]/len(data_temp)).values) # number of individuals in the selected intervention that would make up for a balanced contribution to the covariates
        #print(data_set['group-rct'].value_counts().loc[label])
        print(len(data_new))
        print(n)
        print(p)
        print(label)
        print(data_set['group-rct'].value_counts().loc[label])
        ind_list = data_new.sample(int(n)-int(data_set['group-rct'].value_counts().loc[label])).index.values
        #data_temp = data_new.append(data_set.ix[:, :])
        # Trying to get a more exact sample size.
        #print("ESTE VALOR")        
        #print(n-df['Size'].sum())

    if label == 'control':
        data_new.loc[ind_list,'group-rct'] = "control"
        data_new.loc[set(data_new.index.values ) - set(ind_list),'group-rct'] = "intervention"
        for x in ind_list:
            if (x not in data_new.index):
                print("malo")
                print(x)
    else:
        #data_new['group-rct'] = ["intervention" if x in ind_list else "control" for x in data_new.index]
        data_new.loc[ind_list,'group-rct'] = "intervention"
        data_new.loc[set(data_new.index.values ) - set(ind_list),'group-rct'] = "control"

    #print("Value counts")
    #print(data_new['group-rct'].value_counts())

    todaysdate = str(dt.datetime.today().date())
    data_new['batch'] = int(np.max(data_set.batch.value_counts().index.astype('int').values)) + int(1)

    self.total_data = data_new.append(data_set)
    self.total_data['age'] = age_copy
    self.total_data['date'] = pd.to_datetime(self.total_data['date']).dt.date

    if not self.pure_randomization_boolean: 
        self.name = self.filename1.rsplit("|")[0]+"|"+",".join(self.strat_columns)+'_'+todaysdate+'_'+str(int(len(self.total_data)))+'_'+str(int(int(self.sample_p)))+'_RCT'+'.xlsx'
        self.name_static = self.filename1.rsplit("|")[0]+"|"+",".join(self.strat_columns)+'_'+todaysdate+'_'+str(int(len(self.total_data)))+'_'+str(int(int(self.sample_p)))+'.xlsx'
        #self.total_data['date'] = self.total_data['date'].dt.strftime('%M/%d/%Y')
    else:
        self.name = self.filename1.rsplit("|")[0]+"|"+str(self.pure_randomization_text)+'_'+todaysdate+'_'+str(int(len(self.total_data)))+'_'+str(int(int(self.sample_p)))+'_RCT'+'.xlsx'
        self.name_static = self.filename1.rsplit("|")[0]+"|"+str(self.pure_randomization_text)+'.xlsx'+'_'+todaysdate+'_'+str(int(len(self.total_data)))+'_'+str(int(int(self.sample_p)))+'.xlsx'
 
    print(data_new)

    self.total_data = self.total_data.set_index(self.data_rct.columns[0])

    print(self.total_data)
    data_new.to_excel(self.name_static, na_rep='',index=False)
    self.total_data.to_excel(self.name, na_rep='')

    name_log = self.filename1.rsplit(".")[0]+todaysdate+'_log.xlsx'
    writer = pd.ExcelWriter(name_log, engine = 'xlsxwriter')
    for col in self.strat_columns:
        if col=="age":
            pass
        else:
            pd.crosstab(self.total_data[col], self.total_data['group-rct']).to_excel(writer, sheet_name=col)
    writer.save()

    return self.name

def group_age(df):
    for cols in df.columns:
        if 'age' in cols:
            age_copy = df[cols]
            qtile = df[cols].astype('float').quantile([0.,0.25,0.5,0.75]).values.astype('float')
            df['age'] = df[cols].astype('float')
            df.loc[df['age'] > qtile[len(qtile)-1], 'age'] = '['+str(qtile[len(qtile)-1])+'-'+str(df['age'].max())+']'
            for i in range(len(qtile)-1):
                df.loc[(df['age'] >= qtile[i]) & (df['age']<qtile[i+1]),'age'] = '['+str(qtile[i])+'-'+str(qtile[i+1])+')'
    return df, age_copy, df.index
