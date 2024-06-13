import matplotlib.pyplot as plt

from early_fusion_lstm import LSTMPredictor
from torch.nn.utils import clip_grad_norm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib as mpl
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import time as t
import os
from collections import Counter
from tqdm import tqdm


use_cuda = torch.cuda.is_available()
print('Use CUDA: ' + str(use_cuda))

if use_cuda:
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor


# general model settings
train_batch_size = 128
test_batch_size = 32 # this should stay fixed at 1 when using slow test because the batches are already set in the data loader
prediction_length = 1  # (predict next frame)
sequence_length = 60  # 3s context window

shuffle = False
num_layers = 1

grad_clip_bool = False # turn gradient clipping on or off
grad_clip = 1.0 # try values between 0 and 1
init_std = 0.5
num_epochs = 70
slow_test = True
patience = 10

loss_func = nn.CrossEntropyLoss() # add class weights later to take into account unbalanced data

# set file dir
# input feature dir
annotations_dir = './data/extracted_annotations/bc_mc_labels/'
acous_dir = './data/signals/gemaps_features_processed_50ms/znormalized'
visual_dir = './data/extracted_annotations/visual/manual_50ms'
verbal_dir = './data/extracted_annotations/verbal/0.05'

# file-selection dict
# note here it is used for hyperparameter tuning
#listener_lst = ['Child','Parent','Adult1','Adult2']
listener = 'Adult2'
train_list_path = './data/splits/training_' + listener + '.txt'
validation_list_path = './data/splits/validation_' + listener + '.txt'
file_list_path = './data/splits/'+ listener + '.txt'
file_list = list(pd.read_csv(file_list_path, header=None, dtype=str)[0])
train_file_list = list(pd.read_csv(train_list_path, header=None, dtype=str)[0])
validation_file_list = list(pd.read_csv(validation_list_path, header=None, dtype=str)[0])

# results_dir = './results'
# if not(os.path.exists(results_dir)):
#     os.mkdir(results_dir)


# early feature fusion
def load_data_sliding(file_list, annotations_dir):
    # read files of different modalities
    prediction_length = 1  # !!!
    dataset = list()
    labels = []

    for filename in file_list:
        # load features of different modalities
        vocal = pd.read_csv(acous_dir + '/' + filename + '.csv', delimiter=',')
        visual = pd.read_csv(visual_dir + '/' + filename + '.csv', delimiter=',')
        verbal = pd.read_csv(verbal_dir + '/' + filename + '.csv', delimiter=',')
        min_len_fea = min([len(vocal['frame_time'].tolist()), len(visual['frameTimes'].tolist())
                              , len(verbal['frameTimes'].tolist())])

        x_temp = pd.concat([visual.head(min_len_fea), verbal.head(min_len_fea), vocal.head(min_len_fea)], axis=1)

        temp_y = pd.read_csv(annotations_dir + '/' + filename + '.csv', delimiter=',')
        y_temp = temp_y.head(min_len_fea)

        # remove irrelavent features
        try:
            x_temp.drop('frame_time', axis=1, inplace=True)
        except:
            pass

        try:
            x_temp.drop('frameTimes', axis=1, inplace=True)
        except:
            pass

        # convert into the form of dictionary
        data_x = {}
        for feature_name in x_temp.columns.values.tolist():
            data_x[feature_name] = x_temp[feature_name]

        #### split features into batches
        predict_np = y_temp['val'].tolist()

        num_feat_per_person = {'feature': len(x_temp.columns.values.tolist())}

        # reshape the data to align Roddy's structure
        x_np_dict_list = list()

        # reshape the data structure
        for feature_name in x_temp.columns.values.tolist():
            x_np_dict_list.append(np.squeeze(np.array(data_x[feature_name])))

        x_np_dict = np.asarray(x_np_dict_list).reshape([len(x_np_dict_list), len(x_np_dict_list[0])])

        window = 0

        while (window + sequence_length) < len(predict_np):
            datapoint = {}
            data_temp_x = np.empty([num_feat_per_person['feature'], sequence_length], dtype=np.float32)
            data_temp_x[0:num_feat_per_person['feature'], :] = x_np_dict[:,
                                                               window:window + sequence_length]
            # save info: return training data and labels
            datapoint['x'] = data_temp_x
            datapoint['y'] = predict_np[window + sequence_length]

            dataset.append(datapoint)
            labels.append(datapoint['y'])
            window += 1

    return dataset, labels


train_dataset, train_labels = load_data_sliding(train_file_list, annotations_dir)
test_dataset, test_labels = load_data_sliding(validation_file_list, annotations_dir)
train_count = Counter(train_labels)
test_count = Counter(test_labels)

print("Training sample distribution: {}".format(train_count))
print("Testing sample distribution: {}".format(test_count))

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle, drop_last=True)

num_features = train_dataset[0]['x'].shape[0]

lr = 1e-5
dropout_out = 0.2
L2 = 1e-3

lstm_settings_dict = {
                      'hidden_dims': 50,
                      'layers': 1,
                      'dropout': {'master_out': dropout_out, 'master_in': dropout_out}
                        }


model = LSTMPredictor(lstm_settings_dict=lstm_settings_dict, num_feats=num_features,
                      batch_size=train_batch_size, seq_length=sequence_length, prediction_length=prediction_length)
model.weights_init(init_std)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=L2)

# save the result into a dictionary
results_save = dict()
results_save['train_losses'], results_save['test_losses'], results_save['indiv_perf'], results_save['accuracy_scores'], \
results_save[
    'accuracy_evaluate'], results_save['acc_score_final'], results_save['max_acc'], results_save[
    'min_acc'], results_save['true_vals'], results_save['predicted_vals'] = [], [], [], [], [], [], [], [], [], []

best_acc = 0.0

# %% Training
for epoch in range(0, num_epochs):
    # tell pytorch that you are training the model so that the settings would be different
    model.train()
    t_epoch_strt = t.time()
    loss_list = []
    model.change_batch_size_reset_states(train_batch_size)

    for batch_indx, batch in enumerate(train_dataloader):
        model.init_hidden()
        optimizer.zero_grad()

        # set model input here
        model_input = batch['x'].transpose(1, 2)
        model_output_logits = model(model_input)

        y = batch['y']

        loss = loss_func(model_output_logits, y)
        loss_list.append(loss.cpu().data.numpy())
        loss.backward()

        if grad_clip_bool:
            clip_grad_norm(model.parameters(), grad_clip)

        optimizer.step()


    # stores the CE loss func results
    results_save['train_losses'].append(np.mean(loss_list))

    # %% Test model
    t_epoch_end = t.time()
    model.eval()

    losses_test = list()
    results_dict = dict()
    losses_dict = dict()
    batch_sizes = list()
    predicted_val = list()
    true_val = list()
    losses_l1 = []

    for batch_indx, batch in enumerate(test_dataloader):

        # set model input here
        model_input = batch['x'].transpose(1, 2)
        y_test = batch['y']

        # set the model.change_batch_size directly
        batch_length = 32
        if batch_indx == 0:
            model.change_batch_size_reset_states(batch_length)
        else:
            if slow_test:
                model.change_batch_size_no_reset(batch_length)
            else:
                model.change_batch_size_reset_states(batch_length)

        out_test = model(model_input)

        preds = torch.softmax(out_test, dim=1)
        predictions = np.argmax(preds.data.cpu().numpy(),axis=1)


        # convert 2d to 1d array
        predicted = predictions.flatten()
        true = y_test.data.cpu().numpy().flatten()
        predicted_val.append(predicted)
        true_val.append(true)

        loss = loss_func(out_test, y_test)

        losses_test.append(loss.data.cpu().numpy())
        batch_sizes.append(batch_length)

    # get weighted mean
    # normalize with the batch size(after sigmoid)
    loss_weighted_mean = np.sum(np.array(batch_sizes) * np.squeeze(np.array(losses_test))) / np.sum(batch_sizes)

    # get accuracy after all batches
    true_vals = [item for sublist in true_val for item in sublist]
    predicted_vals = [item for sublist in predicted_val for item in sublist]
    acc_score = accuracy_score(true_vals, predicted_vals)
    if acc_score > best_acc:
        cm = confusion_matrix(true_vals, predicted_vals)
        best_acc = acc_score




    t_total_end = t.time()
    print(
        'Epoch: {0} \t Val_loss: {1}\t Train_Loss: {2} \t Val_acc: {3} \t Train_time: {4} \t Val_time: {5} \t Total_time: {6}'.format(
            epoch + 1,
            np.round(loss_weighted_mean, 4),
            np.round(np.float64(np.array(loss_list).mean()), 4),
            np.around(acc_score, 4),
            np.round(t_epoch_end - t_epoch_strt, 2),
            np.round(t_total_end - t_epoch_end, 2),
            np.round(t_total_end - t_epoch_strt, 2)))

print("Best validation accuracy: {}".format(np.around(best_acc, 4)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

    # if (epoch + 1 > patience) and \
    #         (np.argmin(np.round(results_save['test_losses'], 4)) < (len(results_save['test_losses']) - patience)):
    #     print('early stopping called at epoch: ' + str(epoch + 1))
    #     break
