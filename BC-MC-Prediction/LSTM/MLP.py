# -*- coding: utf-8 -*-
"""
MLP model as the baseline for sequential info

@author: Crystal
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import random
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
encoding = 'utf-8'
from sklearn.metrics import accuracy_score


###########
#load data#
###########

# set file dir
# input feature dir
sequence_length = 40  # 2s context window  
annotations_dir = './data/extracted_annotations/BC_anno'
acous_dir = './data/signals/gemaps_features_processed_50ms/znormalized'
visual_dir = './data/extracted_annotations/visual/manual_50ms'
verbal_dir = './data/extracted_annotations/verbal/0.05'
GS_dir = './data/extracted_annotations/G_surprisal'
LS_dir = './data/extracted_annotations/L_surprisal'
embedding_dir = './data/extracted_annotations/embedding'
results_dir = './results'
if not(os.path.exists(results_dir)):
    os.mkdir(results_dir)

# file-selection dict
# note here it is used for hyperparameter tuning
listener = 'Adult2'
speaker_dict = {'Child':'Parent','Parent':'Child','Adult1':'Adult2','Adult2':'Adult1'}
train_list_path = './data/splits/training_' + listener + '.txt'
validation_list_path = './data/splits/validation_' + listener + '.txt'
file_list_path = './data/splits/'+ listener + '.txt'
file_list = list(pd.read_csv(file_list_path, header=None, dtype=str)[0])
train_file_list = list(pd.read_csv(train_list_path, header=None, dtype=str)[0])
validation_file_list = list(pd.read_csv(validation_list_path, header=None, dtype=str)[0])
utterance_path = './data/annotations/forced_utterance.csv'
utterance_frame = pd.read_csv(utterance_path, dtype=str)
utterance_candi = utterance_frame[utterance_frame['participant']==speaker_dict[listener]] 



def load_data(file_list,annotations_dir,modality):

    # read each file and pad with 0s
    x_all = []
    y_all = []
    
    for filename in validation_file_list:
        
        # get the file list within each folder
        annot_folder = os.fsencode(annotations_dir + '/'+ filename)
        
        x = []
        y = []
        
        for file in os.listdir(annot_folder):
            name = file.decode(encoding)
                
            # load features of different modalities
            vocal_temp = pd.read_csv(acous_dir + '/'+ filename + '/' + name,delimiter=',')  
            vocal_temp = vocal_temp.iloc[: , 1:]
            visual_temp = pd.read_csv(visual_dir + '/'+ filename + '/' + name,delimiter=',')
            visual_temp= visual_temp.iloc[: , 1:] 
            verbal_temp = pd.read_csv(verbal_dir+ '/'+ filename + '/' + name,delimiter=',') 
            verbal_temp= verbal_temp.iloc[: , 1:] 
            GS_temp = pd.read_csv(GS_dir+ '/'+ filename + '/' + name,delimiter=',') 
            GS_temp= GS_temp.iloc[: , 1:] 
            LS_temp = pd.read_csv(LS_dir+ '/'+ filename + '/' + name,delimiter=',') 
            LS_temp= LS_temp.iloc[: , 1:] 
            df = pd.read_csv(annotations_dir+ '/'+ filename + '/' + name,delimiter=',') 
            df = df.iloc[: , 2:] 
            
        
            # loop modalities
            if modality == 'visual':
                temp_x = visual_temp    
            elif modality == 'vocal':
                temp_x = vocal_temp             
            elif modality == 'verbal':
                temp_x = pd.concat([verbal_temp,LS_temp,GS_temp], axis=1)      
                
                    
            elif modality == 'visual_vocal':
                temp_x = pd.concat([visual_temp, vocal_temp], axis=1)
                
            elif modality == 'visual_verbal':
                temp_x = pd.concat([visual_temp,verbal_temp,LS_temp,GS_temp], axis=1)
                
            elif modality == 'verbal_vocal':
                temp_x = pd.concat([verbal_temp,LS_temp,GS_temp,vocal_temp], axis=1)
                
            elif modality == 'all':
                temp_x = pd.concat([visual_temp,verbal_temp,LS_temp,GS_temp,vocal_temp], axis=1)
                
                 
            # reshape the data in the form of looped form so that each frame is equipped with 40 frames' features
            # concatenate 40 frames' context window as the context window
            x_temp = []
            y_temp = []
            
            expand_no = temp_x.shape[0] - sequence_length
            
            iter_no = 0
            while iter_no < expand_no:
                # convert the matrix into a vector
                per_seq = temp_x[iter_no:iter_no + sequence_length].values.tolist() 
                # flatten the nested list
                fea_lst = [item for sublist in per_seq for item in sublist]
                per_seq_y = df[iter_no + sequence_length:iter_no + sequence_length+1].values.tolist()[0][0]
                
                newx = np.array(fea_lst)
                newy = np.array(per_seq_y)
                
                x_temp.append(newx)
                y_temp.append(newy)           
                iter_no += 1
            
            x.append(x_temp)
            y.append(y_temp)    
            
            
        x_all.append(x)
        y_all.append(y)
        
    # flatten the list
    X_data_temp = [item for sublist in x_all for item in sublist]    
    Y_data_temp = [item for sublist in y_all for item in sublist] 
    X_data = [item for sublist in X_data_temp for item in sublist]     
    Y_data = [item for sublist in Y_data_temp for item in sublist]     
    return X_data, Y_data


                     
###############
#run the model#
###############

def run_model(X_train, y_train,X_test,Y_test):
    mlp = MLPClassifier(max_iter=1000)
    parameter_space = {
                    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
                    'activation': ['tanh', 'relu'],
                    'solver': ['sgd', 'adam'],
                    'alpha': [0.0001, 0.05],
                    'learning_rate': ['constant','adaptive'],
                }
        
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    return Y_test, y_pred

################
#run altogether#
################

# LOSO cross-validation
# run the model with the selected parameters 
def run_all(file_list,annotations_dir,modality):
    acc_max_lst = []
    acc_min_lst = []
    
    predicted_tot = []
    true_tot = []
    final_results = []
    n = 0
    for file in file_list:
        # leave one speaker out cross validation
        test_file_list = [file]
        train_file_list = file_list.copy()
        train_file_list.remove(file)
        
        print('Start running fold: ' + str(n))
        X_train, y_train = load_data(train_file_list,annotations_dir,modality)
        X_test,Y_test = load_data(test_file_list,annotations_dir,modality)
        Y_test, y_pred = run_model(X_train, y_train,X_test,Y_test)
        acc_score = accuracy_score(Y_test, y_pred, normalize=True)
        final_results.append(acc_score)
        
        # get the highest accuracy score of test sets for each fold
        acc_max_lst.append(max(final_results))
        acc_min_lst.append(min(final_results))
        
        true_tot.append(Y_test)
        predicted_tot.append(y_pred)
        
        print('Finished running fold: ' + str(n))
        n += 1
    


    # flatten the nested lists
    true_vals = [item for sublist in true_tot for item in sublist]
    predicted_vals = [item for sublist in predicted_tot for item in sublist]
    # true_vals = [item for sublist in true_vals_temp for item in sublist]
    # predicted_vals = [item for sublist in predicted_vals_temp for item in sublist]
    acc_score_final = accuracy_score(true_vals, predicted_vals)
    result = [modality,acc_score_final, final_results, true_vals,predicted_vals]    
    # save more results
    return result

modality_lst = ['verbal','visual','vocal','all','verbal_vocal','visual_verbal']
listener_lst = ['Child','Parent','Adult1','Adult2']  

for modality in modality_lst:  
    for listener in listener_lst:
        result = run_all(file_list,annotations_dir,modality)
    
        output = pd.DataFrame(result)
        name = './result_non-scrambled/' + listener + modality + '.csv'
        output.to_csv(name)
        

trial = load_data(file_list,annotations_dir,'verbal')




for file in file_list:
     # leave one speaker out cross validation
     test_file_list = [file]
     train_file_list = file_list.copy()
     train_file_list.remove(file)
     print(test_file_list)
     print(train_file_list)
     
     
     
     
     
     