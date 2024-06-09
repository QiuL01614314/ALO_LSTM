# -*- coding: utf-8 -*-
"""
Created on Mon May 29 12:24:28 2023

@author: Mahyar Servati
"""
import numpy as np
from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from mealpy.swarm_based import MFO, ALO
import pandas as pd
from keras import optimizers
import time
import os


start_time = time.time()

season = 'spring'

def xy_split(df, target):
    y_clmns=[target]
    x_clmns=df.columns.tolist()
    remove_clmns=[target]
    for arg in remove_clmns:
        x_clmns.remove(arg)
    X=df[x_clmns]
    y=df[y_clmns]
    return X, y    


df = pd.read_csv("DNI.Jiangsu.csv")#, index_col='Time')
# df.index = list(map(lambda x:x.replace("T", " "),df.index))
df['Time'] = list(map(lambda x:x.replace("T", " "),df['Time']))
df.set_index(pd.to_datetime(df['Time']), inplace=True)
df.drop(['Time'], axis=1, inplace=True)

if season == 'spring':
    df2 = df[:2160]
elif season == 'summer':
    df2 = df[2160:4344]
elif season == 'fall':
    df2 = df[4344:6552]
elif season == 'winter':
    df2 = df[6552:]
data = df2.copy()

data.DNI.interpolate(inplace = True)

X, y = xy_split(data, 'DNI')

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2, random_state=1)

X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))


OPT_ENCODER = LabelEncoder()
OPT_ENCODER.fit(['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'])  

WOI_ENCODER = LabelEncoder()
WOI_ENCODER.fit(['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'])

ACT_ENCODER = LabelEncoder()
ACT_ENCODER.fit(['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'])
 
def decode_solution(solution):
    batch_size = 2**int(solution[0])
    epoch = 100 * int(solution[1])
    # opt_integer = int(solution[2])
    # opt = OPT_ENCODER.inverse_transform([opt_integer])[0]
    opt = 'Adam'
    learning_rate = solution[2]
    # network_weight_initial_integer = int(solution[4])
    # network_weight_initial = WOI_ENCODER.inverse_transform([network_weight_initial_integer])[0]
    network_weight_initial = 'normal'
    # act_integer = int(solution[5])
    # activation = ACT_ENCODER.inverse_transform([act_integer])[0]
    activation = 'relu'
    n_hidden_units = 2**int(solution[3])
    return [batch_size, epoch, opt, learning_rate, network_weight_initial, activation, n_hidden_units]


def objective_function(solution): 
    batch_size, epoch, opt, learning_rate, network_weight_initial, Activation, n_hidden_units = decode_solution(solution)
    model = Sequential()
    model.add(LSTM(units = n_hidden_units, activation=Activation,
                   kernel_initializer=network_weight_initial, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units = 1)) 
    optimizer = getattr(optimizers, opt)(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=0)
    yhat = model(X_test)
    fitness = mean_absolute_error(y_test, yhat)
    return fitness

hyper_list = ['batch_size', 'epoch', 'learning_rate', 'n_hidden_units']
LB = [1, 1, 0.0001, 2]
UB = [4.99, 10.99, 1.0, 6.99]
      
problem = {
    'fit_func' : objective_function,
    'lb' : LB,
    'ub' : UB,
    'minmax' : 'min',
    'verbose' : True,
    }

Epoch = 200 #should be in [2, 10000]
Pop_Size = 32 #should be in [10, 10000]

optmodel = ALO.BaseALO(problem, epoch=Epoch, pop_size=Pop_Size)
optmodel.solve()

bests = optmodel.solution[0]


Best_Hyper =  pd.DataFrame(bests, columns=['Best Hyperparameters'], index=hyper_list)


print(Best_Hyper)

end_time = time.time()

print('Optimazing and prediction done in', round(end_time-start_time, 2), "secound")
