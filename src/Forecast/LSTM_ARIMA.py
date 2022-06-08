# -------------------------------------------------------------------------
#                                   Packeterias
# -------------------------------------------------------------------------


import numpy as np
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from datetime import timedelta
import pmdarima as pm

# -------------------------------------------------------------------------
#                                   LSTM FUNCTIONS
# -------------------------------------------------------------------------

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

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

def train_lstm(trainX,trainY, seq_length=4, num_epochs = 2000,learning_rate = 0.01,input_size = 1,hidden_size = 2,num_layers = 1,num_classes = 1):
  lstm = LSTM(seq_length, num_classes, input_size, hidden_size, num_layers)

  criterion = torch.nn.MSELoss()    # mean-squared error for regression
  optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
  #optimizer = torch.optim.SGD(lstm.,(), lr=learning_rate)

  # Train the model
  for epoch in range(num_epochs):
      outputs = lstm(trainX)
      optimizer.zero_grad()
      
      # obtain the loss function
      loss = criterion(outputs, trainY)
      
      loss.backward()
      
      optimizer.step()
      #if epoch % 100 == 0:
        #print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
  return lstm

def predict_future_jojojo(data_last,time_future, window):
  predictions = data_last
  lstm.eval()
  for _ in range(time_future):
    train_predict = lstm(Variable(torch.Tensor(np.array([predictions[-window:]]).reshape(-1,window,1)))).data.numpy()
    predictions.append(train_predict[0][0])
  data_predict = sc.inverse_transform(np.array(predictions).reshape(-1,1))
  return data_predict

def data_train(country, window, time):
  #Filtramos datos
  training_set =cases_who[cases_who.Country ==country][cases_who.New_cases !=0 ]
  sc = MinMaxScaler()
  training_data = sc.fit_transform(training_set.New_cases.values.reshape(-1, 1))
  
  num_layers = 1
  seq_length = window
  x, y = sliding_windows(training_data, seq_length)
  train_size = int(len(y) * 0.80)
  test_size = len(y) - train_size

  #Conjuntos de entrenamiento y prueba
  dataX = Variable(torch.Tensor(np.array(x)))
  dataY = Variable(torch.Tensor(np.array(y)))

  trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
  trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

  testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
  testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
  return trainX, trainY

def plot_ts(time, window, country,type_ts ):
  #Filtramos datos
  training_set =cases_who[cases_who.Country ==country][cases_who[type_ts] !=0 ]
  #sc = MinMaxScaler()
  training_data = sc.fit_transform(training_set[type_ts].values.reshape(-1, 1))
  
  num_layers = 1
  seq_length = window
  x, y = sliding_windows(training_data, seq_length)
  train_size = int(len(y) * 0.80)
  test_size = len(y) - train_size

  #Conjuntos de entrenamiento y prueba
  dataX = Variable(torch.Tensor(np.array(x)))
  dataY = Variable(torch.Tensor(np.array(y)))

  trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
  trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

  testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
  testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

  #Entrenamos
  #lstm = tain_lstm(trainX,trainY)
  lstm.eval()

  train_predict = lstm(dataX)

  data_predict = train_predict.data.numpy()
  dataY_plot = dataY.data.numpy()

  #pred_future = predict_future_jojojo(data_predict[-window:].reshape(1,window)[0].tolist(),time, window)  
  pred_future = predict_future_jojojo(training_data[-window:].reshape(1,window)[0].tolist(),time, window)  

  data_predict = sc.inverse_transform(data_predict)
  dataY_plot = sc.inverse_transform(dataY_plot)

  date_pred = pd.date_range(training_set.Date_reported[-2:].values[0], periods=time)
  
  
  dates_final = np.hstack([training_set.Date_reported, date_pred])
  
  predicts_final = np.vstack([training_set[type_ts][-1:], pred_future])

  
  model_arima = pm.auto_arima(training_set[type_ts], start_p=0, start_q=0,
                    test='kpss',
                    max_p=3, max_q=3,
                    m=1,
                    d=None,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True)
  
  fc = model_arima.predict(n_periods=time, return_conf_int=False)
  
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