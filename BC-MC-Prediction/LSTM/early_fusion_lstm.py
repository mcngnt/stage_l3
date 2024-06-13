import torch
from torch.autograd import Variable
import torch.nn as nn


use_cuda = torch.cuda.is_available()

if use_cuda:
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor


# %% LSTM Class

# lstm axes: [sequence,minibatch,features]
class LSTMPredictor(nn.Module):

    def __init__(self, lstm_settings_dict, num_feats,
                 batch_size=32, seq_length=200, prediction_length=60):
        super(LSTMPredictor, self).__init__()

        # General model settings
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_feats = num_feats
        self.prediction_length = prediction_length

        # lstm_settings_dict
        self.lstm_settings_dict = lstm_settings_dict
        self.num_layers = lstm_settings_dict['layers']

        # Initialize LSTMs

        self.lstm_master = nn.LSTM(self.num_feats,
                                           self.lstm_settings_dict['hidden_dims'], batch_first=True, num_layers=self.num_layers).type(dtype)

        # init dropout layers
        self.dropout_dict = {}
        for drop_key, drop_val in self.lstm_settings_dict['dropout'].items():
            self.dropout_dict[drop_key] = nn.Dropout(drop_val)
            setattr(self, 'dropout_'+str(drop_key),
                    self.dropout_dict[drop_key])

        self.out = nn.Linear(
            self.lstm_settings_dict['hidden_dims'], 3).type(dtype)
        self.out_binary = nn.Linear(self.lstm_settings_dict['hidden_dims'], 1).type(dtype)
        self.init_hidden()

    def init_hidden(self):
        self.hidden_lstm = (Variable(
            torch.zeros(self.num_layers, self.batch_size, self.lstm_settings_dict['hidden_dims'])).type(
            dtype), Variable(torch.zeros(
            self.num_layers, self.batch_size, self.lstm_settings_dict['hidden_dims'])).type(dtype))

    def change_batch_size_reset_states(self, batch_size):
        self.batch_size = int(batch_size)
        self.hidden_lstm = (Variable(
            torch.zeros(self.num_layers, self.batch_size, self.lstm_settings_dict['hidden_dims'])).type(
            dtype), Variable(torch.zeros(
            self.num_layers, self.batch_size, self.lstm_settings_dict['hidden_dims'])).type(dtype))

    def change_batch_size_no_reset(self, new_batch_size):
        self.hidden_lstm = (
                    Variable(
                        self.hidden_lstm[0][:, :new_batch_size, :].data.contiguous().type(dtype)),
                    Variable(self.hidden_lstm[1][:, :new_batch_size, :].data.contiguous().type(dtype)))

        self.batch_size = new_batch_size

    def weights_init(self, init_std):
        # init bias to zero recommended in http://proceedings.mlr.press/v37/jozefowicz15.pdf
        nn.init.normal_(self.out.weight.data, 0, init_std)
        nn.init.constant_(self.out.bias, 0)

        nn.init.normal_(self.lstm_master.weight_hh_l0, 0, init_std)
        nn.init.normal_(self.lstm_master.weight_ih_l0, 0, init_std)
        nn.init.constant_(self.lstm_master.bias_hh_l0, 0)
        nn.init.constant_(self.lstm_master.bias_ih_l0, 0)

    def forward(self, in_data):
        x = in_data
        x = self.dropout_dict['master_in'](x)
        lstm_out, self.hidden_lstm = self.lstm_master(x, self.hidden_lstm)
        lstm_out = self.dropout_dict['master_out'](lstm_out[:,-1,:])

        out = self.out_binary(lstm_out)

        return out
