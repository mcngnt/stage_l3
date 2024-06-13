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
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
import time as t
import os
from collections import Counter, defaultdict
from tqdm import tqdm
import random


use_cuda = torch.cuda.is_available()
print('Use CUDA: ' + str(use_cuda))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if use_cuda:
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor


# general model settings
train_batch_size = 64
test_batch_size = 4 # this should stay fixed at 1 when using slow test because the batches are already set in the data loader
prediction_length = 1  # (predict next frame)
sequence_length = 40  # 2s context window

shuffle = False
num_layers = 1

grad_clip_bool = False # turn gradient clipping on or off
grad_clip = 1.0 # try values between 0 and 1
init_std = 0.84
num_epochs = 59
slow_test = True
patience = 10

lr = 0.0018
dropout_out = 0.2
L2 = 0.00098

lstm_settings_dict = {
                      'hidden_dims': 100,
                      'layers': 3,
                      'dropout': {'master_out': 0.81, 'master_in': 0.11}
                        }

loss_func = nn.CrossEntropyLoss() # add class weights later to take into account unbalanced data

# set file dir
# input feature dir
annotations_dir = './data/extracted_annotations/bc_mc_labels/'
# acous_dir = './data/signals/gemaps_features_processed_50ms/znormalized'
visual_dir = './data/extracted_annotations/visual/manual_50ms'
# verbal_dir = './data/extracted_annotations/verbal/0.05'

# file-selection dict
# note here it is used for hyperparameter tuning
listener_lst = ['Child','Parent','Adult1','Adult2']
# listener_lst = ['Parent']
#train_list_path = './data/splits/training_' + listener + '.txt'
#validation_list_path = './data/splits/validation_' + listener + '.txt'

#train_file_list = list(pd.read_csv(train_list_path, header=None, dtype=str)[0])
#validation_file_list = list(pd.read_csv(validation_list_path, header=None, dtype=str)[0])

# results_dir = './results'
# if not(os.path.exists(results_dir)):
#     os.mkdir(results_dir)


# early feature fusion
def load_data_sliding(file_list, annotations_dir, num_feats=-1):
    # read files of different modalities
    prediction_length = 1  # !!!
    dataset = list()
    dataset_dict = defaultdict(list)

    for filename in file_list:
        # load features of different modalities
        # vocal = pd.read_csv(acous_dir + '/' + filename + '.csv', delimiter=',')
        visual = pd.read_csv(visual_dir + '/' + filename + '.csv', delimiter=',')
        # verbal = pd.read_csv(verbal_dir + '/' + filename + '.csv', delimiter=',')
        # min_len_fea = min([len(vocal['frame_time'].tolist()), len(visual['frameTimes'].tolist())
        #                       , len(verbal['frameTimes'].tolist())])

        # x_temp = pd.concat([visual.head(min_len_fea), verbal.head(min_len_fea), vocal.head(min_len_fea)], axis=1)

        # For one modality
        min_len_fea = len(visual['frameTimes'].tolist())
        x_temp = visual

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
        prev_frame = 100
        count_frame = 1

        while (window + sequence_length) < len(predict_np):
            datapoint = {}
            data_temp_x = np.empty([num_feat_per_person['feature'], sequence_length], dtype=np.float32)
            data_temp_x[0:num_feat_per_person['feature'], :] = x_np_dict[:,
                                                               window:window + sequence_length]
            # save info: return training data and labels
            datapoint['x'] = data_temp_x
            datapoint['y'] = predict_np[window + sequence_length]

            # Get only first 4 frames for each label
            if datapoint['y'] == prev_frame and count_frame > 4:
                window += 1
                continue
            elif datapoint['y'] != prev_frame:
                prev_frame = datapoint['y']
                count_frame = 1

            #dataset.append(datapoint)
            dataset_dict[datapoint['y']].append(datapoint)
            count_frame += 1
            window += 1

    # get equal number of samples for each label
    if num_feats != -1:
        min_samples = num_feats
    else:
        min_samples = min([len(x) for x in dataset_dict.values()])

    for key, values in dataset_dict.items():
        samples = random.sample(values, min_samples)
        dataset.append(samples)

    dataset = [item for x in dataset for item in x]
    labels = [item['y'] for item in dataset]

    return dataset, labels

for listener in listener_lst:

    print("Current listener: {}".format(listener))
    file_list_path = './data/splits/' + listener + '.txt'
    file_list = list(pd.read_csv(file_list_path, header=None, dtype=str)[0])
    fold = 0
    acc_list = []
    f1_weighted_list = []
    train_lens, test_lens = [], []
    for file in file_list:
        # leave one speaker out cross validation
        test_file_list = [file]
        train_file_list = file_list.copy()
        train_file_list.remove(file)
        print('Start running fold: ' + str(fold))

        train_dataset, train_labels = load_data_sliding(train_file_list, annotations_dir)
        test_dataset, test_labels = load_data_sliding(test_file_list, annotations_dir)
        train_lens.append(len(train_labels))
        test_lens.append(len(test_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=shuffle, drop_last=True)
        num_features = train_dataset[0]['x'].shape[0]
        model = LSTMPredictor(lstm_settings_dict=lstm_settings_dict, num_feats=num_features,
                              batch_size=train_batch_size, seq_length=sequence_length,
                              prediction_length=prediction_length)
        model.weights_init(init_std)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=L2)
        best_acc = 0.0
        best_f1_weighted = 0.0

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
                model_output_logits = model(model_input.to(device))

                y = batch['y'].to(device)

                loss = loss_func(model_output_logits, y)
                loss_list.append(loss.cpu().data.numpy())
                loss.backward()

                if grad_clip_bool:
                    clip_grad_norm(model.parameters(), grad_clip)

                optimizer.step()


            # %% Test model
            t_epoch_end = t.time()
            model.eval()

            losses_test = list()
            batch_sizes = list()
            predicted_val = list()
            true_val = list()

            for batch_indx, batch in enumerate(test_dataloader):

                # set model input here
                model_input = batch['x'].transpose(1, 2)
                y_test = batch['y'].to(device)

                # set the model.change_batch_size directly
                batch_length = test_batch_size
                if batch_indx == 0:
                    model.change_batch_size_reset_states(batch_length)
                else:
                    if slow_test:
                        model.change_batch_size_no_reset(batch_length)
                    else:
                        model.change_batch_size_reset_states(batch_length)

                out_test = model(model_input.to(device))

                preds = torch.softmax(out_test, dim=1)
                predictions = np.argmax(preds.data.cpu().numpy(), axis=1)

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
            f1_weighted = f1_score(true_vals, predicted_vals, average='weighted')
            if acc_score > best_acc:
                best_acc = acc_score
                # torch.save(model.state_dict(), "./models/parent_model.pt")
            if f1_weighted > best_f1_weighted:
                best_f1_weighted = f1_weighted

            t_total_end = t.time()
            if (epoch + 1) % 10 == 0:
                print(
                    'Epoch: {0} \t Val_loss: {1}\t Train_Loss: {2} \t Val_acc: {3} \t F1-weighted: {4}  \t Train_time: {5} \t Val_time: {6} \t Total_time: {7}'.format(
                        epoch + 1,
                        np.round(loss_weighted_mean, 4),
                        np.round(np.float64(np.array(loss_list).mean()), 4),
                        np.around(acc_score, 4),
                        np.around(f1_weighted, 4),
                        np.round(t_epoch_end - t_epoch_strt, 2),
                        np.round(t_total_end - t_epoch_end, 2),
                        np.round(t_total_end - t_epoch_strt, 2)))


        print("Best validation accuracy: {}, F1-weighted: {}".format(np.around(best_acc, 4), np.around(best_f1_weighted, 4)))
        acc_list.append(best_acc)
        f1_weighted_list.append(best_f1_weighted)
        fold += 1

    avg_acc = sum(acc_list) / len(acc_list)
    avg_f1_weighted = sum(f1_weighted_list) / len(f1_weighted_list)
    avg_train_len = sum(train_lens) / (3 * len(train_lens))
    avg_test_len = sum(test_lens) / (3 * len(test_lens))
    print("For listener: {}, min_acc: {}, max_acc: {}, avg. acc: {}".format(listener, np.around(min(acc_list), 3), np.around(max(acc_list), 3), np.around(avg_acc, 3)))
    print("For listener: {}, min_F1_weighted: {}, max_F1_weighted: {}, avg. F1_weighted: {}".format(listener, np.around(min(f1_weighted_list), 3), np.around(max(f1_weighted_list), 3), np.around(avg_f1_weighted, 3)))
    print("For listener: {}, avg. train samples per label: {} avg. test samples per label: {}".format(listener, round(avg_train_len), round(avg_test_len)))
