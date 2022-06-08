# -------------------------------------------------------------------------
#                                   Packeterias
# -------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import datetime
from datetime import timedelta
import pmdarima as pm

#Data

cases_who = pd.read_csv('https://raw.githubusercontent.com/ArtemioPadilla/MachineLearningAndCOVID/main/Datasets/SDG-3-Health/WHO-COVID-19-global-data-up.csv')
sc = MinMaxScaler()

# -------------------------------------------------------------------------
#                                   LSTM FUNCTIONS
# -------------------------------------------------------------------------


class LSTM(nn.Module):

    def __init__(self, seq_length, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out


def predict_future_jojojo(model,data_last,time_future, window):
  predictions = data_last
  model.eval()
  for _ in range(time_future):
    train_predict = model(Variable(torch.Tensor(np.array([predictions[-window:]]).reshape(-1,window,1)))).data.numpy()
    predictions.append(train_predict[0][0])
  data_predict = sc.inverse_transform(np.array(predictions).reshape(-1,1))
  return data_predict


def plot_ts(model,arima_path,time, window, country,type_ts ):
  #Filtramos datos
  training_set =cases_who[cases_who.Country ==country][cases_who[type_ts] !=0 ]
  #sc = MinMaxScaler()
  training_data = sc.fit_transform(training_set[type_ts].values.reshape(-1, 1))
  
  num_layers = 1
  seq_length = window
  
  #Entrenamos
  #lstm = tain_lstm(trainX,trainY)
  model.eval()
  
  #pred_future = predict_future_jojojo(data_predict[-window:].reshape(1,window)[0].tolist(),time, window)  
  pred_future = predict_future_jojojo(model,training_data[-window:].reshape(1,window)[0].tolist(),time, window)  

  date_pred = pd.date_range(training_set.Date_reported[-2:].values[0], periods=time)
  
  with open(arima_path, 'rb') as pkl:
    fc = pickle.load(pkl).predict(n_periods=time, return_conf_int=False)
  
  fig = go.Figure()

  # Add traces
  fig.add_trace(go.Scatter(x = training_set.Date_reported, y=training_set[type_ts],
                      mode='lines',
                      name='Observaciones reales'))
  fig.add_trace(go.Scatter(x=date_pred.astype('str')[1:], y=pred_future.reshape(1, len(pred_future))[0],
                      mode='lines+markers',
                      name='Predicciones con LSTM'))
  fig.add_trace(go.Scatter(x=date_pred.astype('str')[1:], y=fc,
                      mode='lines+markers',
                      name='Predicciones con ARIMA'))
  
  type_ts_title = type_ts.replace("_", " ")
  fig.update_layout(title= type_ts_title + ' confirmed in ' + country,
                  yaxis_zeroline=False, xaxis_zeroline=False)

  return fig