
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM model for BC prediction

@author: Crystal
"""

from lstm_model import LSTMPredictor
from torch.nn.utils import clip_grad_norm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from os import mkdir
from os.path import exists
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
import time as t
import pickle
import platform
import os
import distro 
from itertools import islice
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import optuna
from optuna.trial import TrialState

###############
#Basic setting#
###############

# set cuda usage
plat = linux_distro = distro.like()   # change the above line to this due to the python version 
my_node = platform.node()
use_cuda = torch.cuda.is_available()
print('Use CUDA: ' + str(use_cuda))

if use_cuda:
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
    p_memory = True
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor
    p_memory = True
    
# general model settings
train_batch_size = 128
test_batch_size = 1 # this should stay fixed at 1 when using slow test because the batches are already set in the data loader
prediction_length = 1  # (predict next frame)  
sequence_length = 60  # 3s context window  
alpha = 0.99  # smoothing constant
init_std = 0.5
momentum = 0

shuffle = False
num_layers = 1
onset_test_flag = True
no_subnets = True

freeze_glove_embeddings = False
grad_clip_bool = False # turn gradient clipping on or off
grad_clip = 1.0 # try values between 0 and 1
init_std = 0.5
num_epochs = 1500
slow_test = True
early_stopping = True
patience = 10
use_date_str = True
hidden_nodes_master = 50
hidden_nodes_acous = 50
hidden_nodes_visual = 0

# set loss functions
loss_func_L1 = nn.L1Loss()
loss_func_L1_no_reduce = nn.L1Loss(reduce=False)
# add the pos_weight based on the proportion
pos_weight = torch.full([1], 7)
loss_func_BCE = torch.nn.BCELoss()
# add sigmoid activation internally
loss_func_BCE_Logit = nn.BCEWithLogitsLoss(pos_weight = pos_weight)

# set file dir
# input feature dir
annotations_dir = './data/extracted_annotations/voice_activity/'
acous_dir = './data/signals/gemaps_features_processed_50ms/znormalized'
visual_dir = './data/extracted_annotations/visual/manual_50ms'
verbal_dir = './data/extracted_annotations/verbal/0.05'

# file-selection dict
# note here it is used for hyperparameter tuning
#listener_lst = ['Child','Parent','Adult1','Adult2']
listener = 'Child'
train_list_path = './data/splits/training_' + listener + '.txt'
validation_list_path = './data/splits/validation_' + listener + '.txt'
file_list_path = './data/splits/'+ listener + '.txt'
file_list = list(pd.read_csv(file_list_path, header=None, dtype=str)[0])
train_file_list = list(pd.read_csv(train_list_path, header=None, dtype=str)[0])
validation_file_list = list(pd.read_csv(validation_list_path, header=None, dtype=str)[0])

results_dir = './results'
if not(os.path.exists(results_dir)):
    os.mkdir(results_dir)


def perf_plot(results_save, results_key, result_dir_name,modality):
    # results_dict, dict_key
    plt.figure()
    plt.plot(results_save[results_key])
    p_max = np.round(np.max(np.array(results_save[results_key])), 4)
    p_min = np.round(np.min(np.array(results_save[results_key])), 4)
    #    p_last = np.round(results_save[results_key][-1],4)
    plt.annotate(str(p_max), (np.argmax(np.array(results_save[results_key])), p_max))
    plt.annotate(str(p_min), (np.argmin(np.array(results_save[results_key])), p_min))
    #    plt.annotate(str(p_last), (len(results_save[results_key])-1,p_last))
    plt.title(modality, fontsize=6)
    plt.xlabel('epoch')
    plt.ylabel(results_key)
    plt.savefig(results_dir + '/' + result_dir_name + '/' + results_key + '.pdf')


def get_chunk(results_dict,true_vals,predicted_vals,label):
    length_to_split = []
    for key, value in results_dict.items():
        length = value.size
        length_to_split.append(length)
    Input = iter(true_vals)
    Output = [list(islice(Input, elem)) for elem in length_to_split]
    Input_pre = iter(predicted_vals)
    Output_pre = [list(islice(Input_pre, elem)) for elem in length_to_split]
    Filenames = [*results_dict]
    # get the indices of BC
    length_lst = []  
    index_lst = []
    
    final = pd.DataFrame() 
    m = 0
    while m < len(Output):
        values = np.array(Output[m])
        indices = np.where(values == label)[0]  
        # get the length of BC chunks
        nums = indices.tolist()
        ranges = sum((list(t) for t in zip(nums, nums[1:]) if t[0]+1 != t[1]), [])
        iranges = iter(nums[0:1] + ranges + nums[-1:])
        temp = []
        for n in iranges:
            length = next(iranges) - n + 1           
            temp.append(length)
       
        # summarize the length of BC
        length_lst.append(Counter(temp))
        nums_new = iter(nums)
        index = [list(islice(nums_new, elem)) for elem in temp]
        index_lst.append(index)
        
        # get the dataframe
        df = pd.DataFrame(list(temp),columns = ['Length'])
        # add the index as another column
        df['Indices'] = index
        df['Filename'] = Filenames[m]
        final = pd.concat([final,df])
        #final.reset_index(drop=True, inplace=True)
        m += 1
        
    return Output,Output_pre,final


def negative_sample(results_dict,true_vals,predicted_vals):
    true_vals_seg,predicted_vals_seg,BC_frame = get_chunk(results_dict,true_vals,predicted_vals,1)
    true_vals_seg,predicted_vals_seg,nonBC_frame = get_chunk(results_dict,true_vals,predicted_vals,0)
    
    final = pd.DataFrame() 
    Filenames = [*results_dict]
    for filename in Filenames:
        # sort BC and non-BC to make full use of each sequence
        BC_candi = BC_frame.loc[BC_frame['Filename']==filename].sort_values(by='Length', ascending=True).reset_index()  
        nonBC_candi = nonBC_frame.loc[nonBC_frame['Filename']==filename].sort_values(by='Length', ascending=True).reset_index()  
        n = 0
        matched = []    
        try:    
            while n < BC_candi.shape[0]:
                # zoom into the same conversation file
                # match once for each sequence chunk
                try:   
                    # put inside loop bc this is influenced by the index change in nonBC_candi
                    matched_frame = nonBC_candi.loc[nonBC_candi['Length'] >= BC_candi['Length'][n]] 
                    # rank the candidate based on length
                    matched_frame.sort_values(by='Length', ascending=True)                       
                    # get the corresponding length of indices(sample from beginnning)
                    matched_index = matched_frame['Indices'].tolist()[0][:BC_candi['Length'][n]] 
                    
                    row_index = matched_frame.index[0]
                    # remove the target line in case of overlapping indices
                    nonBC_candi = nonBC_candi.drop(row_index)
                    matched.append(matched_index)
                    
                # multiple sampling for some super long non-BC instances
                except:                   
                    # get the corresponding length of indices
                    # sample reversely
                    nonBC_candi1 = nonBC_frame.loc[nonBC_frame['Filename']==filename].sort_values(by='Length', ascending=True)  
                    matched_frame = nonBC_candi1.loc[nonBC_candi1['Length'] >= BC_candi['Length'][n]] 
                    # rank the candidate based on length
                    matched_frame.sort_values(by='Length', ascending=True)                       
                    # get the corresponding length of indices(sample from beginnning)
                    matched_index = matched_frame['Indices'].tolist()[0][-BC_candi['Length'][n]:]  
                    matched.append(matched_index)
                    
                temp = BC_candi
                n += 1
            temp['selected'] = matched
            final = pd.concat([final,temp])
        except:
            print(filename)
    return true_vals_seg,predicted_vals_seg,final


# early feature fusion  
def load_data(file_list,annotations_dir,modality):   
    # read files of different modalities    
    results_lengths = dict()
    prediction_length = 1   #!!!
    dataset = list()
    
           
    for filename in file_list:
        # load features of different modalities
        vocal = pd.read_csv(acous_dir+'/'+filename+'.csv',delimiter=',')  
        visual = pd.read_csv(visual_dir+'/'+filename+'.csv',delimiter=',') 
        verbal = pd.read_csv(verbal_dir+'/'+filename+'.csv',delimiter=',') 
        min_len_fea = min([len(vocal['frame_time'].tolist()),len(visual['frameTimes'].tolist())
                           ,len(verbal['frameTimes'].tolist())])
        
        # loop modalities
        if modality == 'visual':
            temp_x = visual.head(min_len_fea)       
        elif modality == 'vocal':
            temp_x = vocal.head(min_len_fea)              
        elif modality == 'verbal':
            temp_x = verbal.head(min_len_fea)         
            
                
        elif modality == 'visual_vocal':
            df_concat = pd.concat([visual.head(min_len_fea), vocal.head(min_len_fea)], axis=1)
            temp_x = df_concat.T.drop_duplicates().T
        elif modality == 'visual_verbal':
            df_concat = pd.concat([visual.head(min_len_fea),verbal.head(min_len_fea)], axis=1)
            temp_x = df_concat.T.drop_duplicates().T
        elif modality == 'verbal_vocal':
            df_concat = pd.concat([verbal.head(min_len_fea),vocal.head(min_len_fea)], axis=1)
            temp_x = df_concat.T.drop_duplicates().T
        elif modality == 'all':
            df_concat = pd.concat([visual.head(min_len_fea),verbal.head(min_len_fea),vocal.head(min_len_fea)], axis=1)
            temp_x = df_concat.T.drop_duplicates().T        


        temp_y = pd.read_csv(annotations_dir+'/'+filename+'.csv',delimiter=',')          
        #min_len = min([len(temp_x['frame_time'].tolist()),len(temp_y['frameTimes'].tolist())])
        
        # truncate two dataframes
        #x_temp = temp_x.head(min_len_fea)
        x_temp = temp_x
        y_temp = temp_y.head(min_len_fea)
        # convert into the form of dictionary
        
        data_x = {}
        for feature_name in x_temp.columns.values.tolist():         
            data_x[feature_name] = x_temp[feature_name]
          
        #### split features into batches 
        file_dur = len(y_temp['val'].tolist())
        num_batches = int(np.floor(file_dur/sequence_length))
        results_lengths[filename] = num_batches * sequence_length
        #data_np_times = np.array(y_temp['frameTimes']) 
        predict_np = np.array([np.roll(y_temp['val'],-roll_indx) for roll_indx in range(1,prediction_length+1) ]).transpose()
        
        # remove irrelavent features
        try:
            x_temp.drop('frame_time', axis=1, inplace=True)
        except:
            pass
        
        try:
            x_temp.drop('frameTimes', axis=1, inplace=True)
        except:
            pass
        
        y_temp.drop('frameTimes', axis=1, inplace=True)
        
        
        num_feat_per_person = {'feature':len(x_temp.columns.values.tolist())}
        
        # reshape the data to align Roddy's structure
        x_np_dict_list = list()
        x_np_dict = list()
        # reshape the data structure   
        for feature_name in x_temp.columns.values.tolist():
            x_np_dict_list.append(np.squeeze(np.array(data_x[feature_name]))) 
        
      
        x_np_dict = np.asarray(x_np_dict_list).reshape([len(x_np_dict_list),len(x_np_dict_list[0])])
        
        for i in range(1,num_batches + 1):
            datapoint = {}
            data_temp_x = []   
            data_temp_x = np.empty([num_feat_per_person['feature'],sequence_length],dtype=np.float32)
            data_temp_x[0:num_feat_per_person['feature'],:] = x_np_dict[:,(i-1)*sequence_length:i*sequence_length]
              
            # save info: return training data and labels
            datapoint['x'] = data_temp_x 
            datapoint['y'] = predict_np[(i-1)*sequence_length:i*sequence_length]
            
            dataset.append(datapoint)
        
    return dataset, results_lengths

def test(model,test_dataset,test_file_list,test_dataloader,results_save,test_results_length):
    losses_test = list()
    results_dict = dict()
    losses_dict = dict()
    batch_sizes = list()
    predicted_val = list()
    true_val = list()
    losses_l1 = []
    model.eval()
    # setup results_dict
    
    for file_name in test_file_list:
        
        # create new arrays for the results
        results_dict[file_name] = np.zeros([test_results_length[file_name], prediction_length])
        losses_dict[file_name] = np.zeros([test_results_length[file_name], prediction_length])
    
    for batch_indx, batch in enumerate(test_dataloader):
       
        # set model input here
        
        model_input_temp = batch['x'].transpose(1, 2).transpose(0, 1)
        model_input = [model_input_temp,[],[],[]]
        
        y_test = batch['y'].transpose(0, 1)
        
    
        # set the model.change_batch_size directly
        batch_length = 1
        if batch_indx == 0:
            model.change_batch_size_reset_states(batch_length)
        else:
            if slow_test:
                model.change_batch_size_no_reset(batch_length)
            else:
                model.change_batch_size_reset_states(batch_length)

        out_test = model(model_input)
        
        out_test = torch.transpose(out_test, 0, 1)
        
        
        # convert to binary class to prepare for f-scores        
        threshold = torch.tensor([0.5])
        predicted = ((out_test>threshold).float()*1).numpy()
        true = ((y_test.transpose(0, 1)>threshold).float()*1).numpy()
        # convert 2d to 1d array
        predicted = predicted.flatten()
        true = true.flatten()
        predicted_val.append(predicted)
        true_val.append(true)
        
        
        
        # Should be able to make other loss calculations faster
        
        loss = loss_func_BCE(F.sigmoid(out_test), y_test.transpose(0,1).float())
        # loss = loss_func_BCE_Logit(out_test,y_test.transpose(0,1))
        
        losses_test.append(loss.data.cpu().numpy())
        batch_sizes.append(batch_length)

        loss_l1 = loss_func_L1(out_test, y_test.transpose(0, 1))
        losses_l1.append(loss_l1.data.cpu().numpy())
        
        
    # get weighted mean
    # normalize with the batch size(after sigmoid)
    loss_weighted_mean = np.sum(np.array(batch_sizes) * np.squeeze(np.array(losses_test))) / np.sum(batch_sizes)
    
    
    # get accuracy after all batches
    true_vals = [item for sublist in true_val  for item in sublist]
    predicted_vals = [item for sublist in predicted_val for item in sublist]
    acc_score = accuracy_score(true_vals, predicted_vals, normalize=True)
    
    # evaluate the model based on equivalent no. of BC and non-BC
    
    true_vals_seg,predicted_vals_seg,info_frame = negative_sample(results_dict,true_vals,predicted_vals)
    # get the values based on indices 
    Filenames = [*results_dict]
    n = 0
    true_final = []
    predict_final = []
    while n<len(Filenames):
        
        index_BC = info_frame.loc[info_frame['Filename']==Filenames[n]]['Indices'].tolist()
        index_BC = [item for sublist in index_BC for item in sublist]
        index_nonBC = info_frame.loc[info_frame['Filename']==Filenames[n]]['selected'].tolist()
        index_nonBC = [item for sublist in index_nonBC for item in sublist]
        temp=[index_BC,index_nonBC]
        flat_list = [item for sublist in temp for item in sublist]
        # convert index into actual results
        true_lst = []
        predict_lst = []
        for index in flat_list:
            true_lst.append(true_vals_seg[n][index])
            predict_lst.append(predicted_vals_seg[n][index])
        true_final.append(true_lst)
        predict_final.append(predict_lst)
        n+=1
      # flatten the lists
    true_label = [item for sublist in true_final for item in sublist]
    predict_label = [item for sublist in predict_final for item in sublist]   
    accuracy_evaluate = accuracy_score(true_label, predict_label, normalize=True)  
             

    results_save['test_losses'].append(loss_weighted_mean)
    results_save['accuracy_scores'].append(acc_score)
    results_save['accuracy_evaluate'].append(accuracy_evaluate)
    results_save['true_vals'].append(true_vals)
    results_save['predicted_vals'].append(predicted_vals) 
     
    return results_save,model_input


def run_model(para_results,train_file_list, test_file_list,modality):
    # load files
   
    
    train_dataset, train_results_length = load_data(train_file_list,annotations_dir,modality)               
    test_dataset, test_results_length = load_data(test_file_list,annotations_dir,modality) 


    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle, num_workers=0,
                                      drop_last=True, pin_memory=p_memory)

    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle, num_workers=0,
                                      drop_last=True, pin_memory=p_memory)
     
    feature_size_dict = {}
    feature_size_dict['acous'] = train_dataset[0]['x'].shape[0]
   
    
    #embedding_info = train_dataset.get_embedding_info()
    embedding_info = {'acous': [], 'visual': []}
   
    # tune the drop-out; L2 
    
    lstm_settings_dict = {'no_subnets': True, 
     'hidden_dims': {'master': 50, 'acous': 50, 'visual': 0}, 
     'uses_master_time_rate': {'acous': True}, 
     'time_step_size': {'acous': 1}, 
     'is_irregular': {'acous': False}, 
     'layers': 1, 
     'dropout': {'master_out': para_results['dropout_out'], 'master_in': para_results['dropout_out'], 'acous_in': 0.25, 'acous_out': 0.25, 'visual_in': 0, 'visual_out': 0.0}, 
     'active_modalities': ['acous']}

    l2_dict = {
        'emb': 0.0001,
        'out': para_results['dropout_out'],
        'master': 0.00001,
        'acous': para_results['dropout_out'],
        'visual': 0.}
     
    model = LSTMPredictor(lstm_settings_dict=lstm_settings_dict, feature_size_dict=feature_size_dict,
                          batch_size=train_batch_size, seq_length=sequence_length, prediction_length=prediction_length,
                          embedding_info=embedding_info)
    
    model.weights_init(init_std)
    
    optimizer_list = []

    optimizer_list.append( optim.Adam( model.out.parameters(), lr=para_results['lr'], weight_decay=l2_dict['out'] ) )
    for embed_inf in embedding_info.keys():
        if embedding_info[embed_inf]:
            for embedder in embedding_info[embed_inf]:
                if embedder['embedding_use_func'] or (embedder['use_glove'] and not(lstm_settings_dict['freeze_glove'])):
                    optimizer_list.append(
                        optim.Adam( model.embedding_func.parameters(), lr=para_results['lr'], weight_decay=l2_dict['emb'] )
                                          )

    for lstm_key in model.lstm_dict.keys():
        optimizer_list.append(optim.Adam(model.lstm_dict[lstm_key].parameters(), lr=para_results['lr'], weight_decay=l2_dict[lstm_key]))

    
    # save the result into a dictionary
    results_save = dict()
    results_save['train_losses'], results_save['test_losses'], results_save['indiv_perf'], results_save['accuracy_scores'],results_save[
        'accuracy_evaluate'], results_save['acc_score_final'],results_save['max_acc'],results_save[
            'min_acc'],results_save['true_vals'],results_save['predicted_vals']= [], [], [], [], [], [], [], [], [], []
 
    
    # %% Training
    for epoch in range(0, num_epochs):
        # tell pytorch that you are training the model so that the settings would be different
        model.train()
        t_epoch_strt = t.time()
        loss_list = []
        model.change_batch_size_reset_states(train_batch_size)
    
        for batch_indx, batch in enumerate(train_dataloader):
            # b should be of form: (x,x_i,v,v_i,y,info)
            model.init_hidden()
            model.zero_grad()
            model_input = []
    
            # set model input here
            
            model_input_temp = batch['x'].transpose(1, 2).transpose(0, 1)
            model_input = [model_input_temp,[],[],[]]
            
            
            model_output_logits = model(model_input)
       
            y = batch['y'].transpose(0, 1)
            
            loss = loss_func_BCE_Logit(model_output_logits,y.float())
            loss_list.append(loss.cpu().data.numpy())
            loss.backward()
            
            if grad_clip_bool:
                clip_grad_norm(model.parameters(), grad_clip)
            
            for opt in optimizer_list:
                opt.step()
        
        # stores the BCE loss func results
        results_save['train_losses'].append(np.mean(loss_list))
        
        
        # %% Test model
        t_epoch_end = t.time()
        model.eval()
        results_save, test_input = test(model,test_dataset,test_file_list,test_dataloader,results_save, test_results_length)
        model.train()
        t_total_end = t.time()
        #        torch.save(model,)
        print(
            '{0} \t Test_loss: {1}\t Train_Loss: {2} \t Acc: {3} \t Acc_evaluate: {4}\t Train_time: {5} \t Test_time: {6} \t Total_time: {7}'.format(
                epoch + 1,
                np.round(results_save['test_losses'][-1], 4),
                np.round(np.float64(np.array(loss_list).mean()), 4),
                np.around(results_save['accuracy_scores'][-1], 4),
                np.around(results_save['accuracy_evaluate'][-1], 4),
                np.round(t_epoch_end - t_epoch_strt, 2),
                np.round(t_total_end - t_epoch_end, 2),
                np.round(t_total_end - t_epoch_strt, 2)))
        
        
        if (epoch + 1 > patience) and \
                (np.argmin(np.round(results_save['test_losses'], 4)) < (len(results_save['test_losses']) - patience)):
            print('early stopping called at epoch: ' + str(epoch + 1))
            break
    
    return results_save,model

def plot_results(results_save, model,modality):
    # save all the results
    result_dir_name = t.strftime('%Y%m%d%H%M%S')[5:-2]
    result_dir_name = modality + result_dir_name + modality
        
    if not (exists(results_dir)):
        mkdir(results_dir)
        
    if not (exists(results_dir + '/' + result_dir_name)):
        mkdir(results_dir + '/' + result_dir_name)
     
    # save results and model parameters
    pickle.dump(results_save, open(results_dir + '/' + result_dir_name + '/results.p', 'wb'))
    torch.save(model.state_dict(), results_dir + '/' + result_dir_name + '/model.p')
    
    perf_plot(results_save, 'train_losses', result_dir_name)
    perf_plot(results_save, 'test_losses', result_dir_name)
    perf_plot(results_save, 'accuracy_scores', result_dir_name) 
    perf_plot(results_save, 'accuracy_evaluate', result_dir_name) 
    plt.close('all')

# use train and validation set to get the optimal parameters  
# loop listenrs and modalities
modality_lst = ['visual','vocal','verbal'] 
for modality in modality_lst:   
    def objective(trial):
        # load files
        
        train_dataset, train_results_length = load_data(train_file_list,annotations_dir,modality)               
        test_dataset, test_results_length = load_data(validation_file_list,annotations_dir,modality) 
    
    
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle, num_workers=0,
                                          drop_last=True, pin_memory=p_memory)
    
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle, num_workers=0,
                                          drop_last=True, pin_memory=p_memory)
         
        feature_size_dict = {}
        feature_size_dict['acous'] = train_dataset[0]['x'].shape[0]
       
        
        #embedding_info = train_dataset.get_embedding_info()
        embedding_info = {'acous': [], 'visual': []}
       
        # tune the drop-out; L2 
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        dropout_out = trial.suggest_float("dropout_out",0.000001, 0.5)
        L2 = trial.suggest_float("L2",0, 1e-4)
        
        lstm_settings_dict = {'no_subnets': True, 
         'hidden_dims': {'master': 50, 'acous': 50, 'visual': 0}, 
         'uses_master_time_rate': {'acous': True}, 
         'time_step_size': {'acous': 1}, 
         'is_irregular': {'acous': False}, 
         'layers': 1, 
         'dropout': {'master_out': dropout_out, 'master_in': dropout_out, 'acous_in': 0.25, 'acous_out': 0.25, 'visual_in': 0, 'visual_out': 0.0}, 
         'active_modalities': ['acous']}
    
        l2_dict = {
            'emb': 0.0001,
            'out': L2,
            'master': 0.00001,
            'acous': L2,
            'visual': 0.}
         
        model = LSTMPredictor(lstm_settings_dict=lstm_settings_dict, feature_size_dict=feature_size_dict,
                              batch_size=train_batch_size, seq_length=sequence_length, prediction_length=prediction_length,
                              embedding_info=embedding_info)
        
        model.weights_init(init_std)
        
        optimizer_list = []
    
        optimizer_list.append( optim.Adam( model.out.parameters(), lr=lr, weight_decay=l2_dict['out'] ) )
        for embed_inf in embedding_info.keys():
            if embedding_info[embed_inf]:
                for embedder in embedding_info[embed_inf]:
                    if embedder['embedding_use_func'] or (embedder['use_glove'] and not(lstm_settings_dict['freeze_glove'])):
                        optimizer_list.append(
                            optim.Adam( model.embedding_func.parameters(), lr=lr, weight_decay=l2_dict['emb'] )
                                              )
    
        for lstm_key in model.lstm_dict.keys():
            optimizer_list.append(optim.Adam(model.lstm_dict[lstm_key].parameters(), lr=lr, weight_decay=l2_dict[lstm_key]))
    
        
        # save the result into a dictionary
        results_save = dict()
        results_save['train_losses'], results_save['test_losses'], results_save['indiv_perf'], results_save['accuracy_scores'],results_save[
            'accuracy_evaluate'], results_save['acc_score_final'],results_save['max_acc'],results_save[
                'min_acc'],results_save['true_vals'],results_save['predicted_vals']= [], [], [], [], [], [], [], [], [], []
     
        
        # %% Training
        for epoch in range(0, num_epochs):
            # tell pytorch that you are training the model so that the settings would be different
            model.train()
            t_epoch_strt = t.time()
            loss_list = []
            model.change_batch_size_reset_states(train_batch_size)
        
            for batch_indx, batch in enumerate(train_dataloader):
                # b should be of form: (x,x_i,v,v_i,y,info)
                model.init_hidden()
                model.zero_grad()
                model_input = []
        
                # set model input here
                
                model_input_temp = batch['x'].transpose(1, 2).transpose(0, 1)
                model_input = [model_input_temp,[],[],[]]
                
                #y = Variable(batch[4].type(dtype).transpose(0, 2).transpose(1, 2))
                
                model_output_logits = model(model_input)
           
                y = batch['y'].transpose(0, 1)
                
                loss = loss_func_BCE_Logit(model_output_logits,y.float())
                loss_list.append(loss.cpu().data.numpy())
                loss.backward()
                
                if grad_clip_bool:
                    clip_grad_norm(model.parameters(), grad_clip)
                
                for opt in optimizer_list:
                    opt.step()
            
            # stores the BCE loss func results
            results_save['train_losses'].append(np.mean(loss_list))
            
            
            # %% Test model
            t_epoch_end = t.time()
            model.eval()
            results_save, test_input = test(model,test_dataset,validation_file_list,test_dataloader,results_save, test_results_length)
            model.train()
            t_total_end = t.time()
            #        torch.save(model,)
            print(
                '{0} \t Val_loss: {1}\t Train_Loss: {2} \t Acc: {3} \t Acc_evaluate: {4}\t Train_time: {5} \t Val_time: {6} \t Total_time: {7}'.format(
                    epoch + 1,
                    np.round(results_save['test_losses'][-1], 4),
                    np.round(np.float64(np.array(loss_list).mean()), 4),
                    np.around(results_save['accuracy_scores'][-1], 4),
                    np.around(results_save['accuracy_evaluate'][-1], 4),
                    np.round(t_epoch_end - t_epoch_strt, 2),
                    np.round(t_total_end - t_epoch_end, 2),
                    np.round(t_total_end - t_epoch_strt, 2)))
            
            
            
            if (epoch + 1 > patience) and \
                    (np.argmin(np.round(results_save['test_losses'], 4)) < (len(results_save['test_losses']) - patience)):
                print('early stopping called at epoch: ' + str(epoch + 1))
                break
        
        accuracy = max(results_save['accuracy_scores'])   
        trial.report(accuracy, epoch)
    
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return accuracy


    def run_tuned(file_list,modality):
        para_results = {}
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100, timeout=600)
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        print("Study statistics: ")
        print("Number of finished trials: ", len(study.trials))
        print("Number of pruned trials: ", len(pruned_trials))
        print("Number of complete trials: ", len(complete_trials))
        
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
            
        # save the best combinations of parameters in the form of dictionary
        para_results["Number of finished trials"] = len(study.trials)
        para_results["Number of pruned trials"] = len(pruned_trials)
        para_results["Number of complete trials"] = len(complete_trials)
        param = trial.params
        para_results.update(param)
        
        # LOSO cross-validation
        # run the model with the selected parameters 
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
            
            results,model = run_model(para_results,train_file_list, test_file_list,modality)
            final_results.append(results)
            
            # get the highest accuracy score of test sets for each fold
            acc_max_lst.append(max(results['accuracy_scores'][-(round(len(results['accuracy_scores'])/10)):]))
            acc_min_lst.append(min(results['accuracy_scores'][-(round(len(results['accuracy_scores'])/10)):]))
            true_tot.append(results['true_vals'])
            predicted_tot.append(results['predicted_vals'])
            
            print('Finished running fold' + str(n))
            n += 1
        
    
    
        # flatten the nested lists
        true_vals_temp = [item for sublist in true_tot for item in sublist]
        predicted_vals_temp = [item for sublist in predicted_tot for item in sublist]
        true_vals = [item for sublist in true_vals_temp for item in sublist]
        predicted_vals = [item for sublist in predicted_vals_temp for item in sublist]
        acc_score_final = accuracy_score(true_vals, predicted_vals)
        
        # choose the structure/parameter information based on each fold's f-scores
        max_acc = max(acc_max_lst) 
        max_index = acc_max_lst.index(max_acc)
        min_acc = min(acc_min_lst) 
        results_save = final_results[max_index]
        # save more results
        results_save['acc_score_final'] = acc_score_final
        results_save['max_acc'] = max_acc
        results_save['min_acc'] = min_acc
    
        # save and plot final results
        plot_results(results_save, model,modality)
        return results_save

    run_tuned(file_list,modality)








        




