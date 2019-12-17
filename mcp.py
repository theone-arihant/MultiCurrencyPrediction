import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.utils import normalize
from keras.models import Sequential, load_model
from keras.layers import Bidirectional, Dense, LSTM, SimpleRNN, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import SGD,Adagrad
from keras import regularizers
from sklearn.model_selection import train_test_split
import multiprocessing as mp
from multiprocessing import Pool
import tensorflow as tf
import keras.backend as K


#tensorflow 
session_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
tf.set_random_seed(1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

early_stop = EarlyStopping(monitor='loss', patience=5, verbose=2)

def adj_r2_score(r2, n, k):
    return 1-((1-r2)*((n-1)/(n-k-1)))


def mcp(df):
    curruncies = ['EUR','JPY','GBP','AUD','CAD','CHF','CNY','SEK','NZD','INR']
    df_temp = df
    print(df.describe())
    df.describe().to_csv('desc.csv')
    print(df.corr(method='spearman'))
    sns.heatmap(df.corr(method='kendall'))
    plt.show()
    df.corr().to_csv('Currency Correlation_spearman.csv')
    
    df['Date'] = pd.to_datetime(df["Date"])
    df_idx = df.set_index(["Date"], drop=True)
    df_idx = df_idx.sort_index(axis=1, ascending=True)
    df_idx = df_idx.iloc[::-1]

    for cur in curruncies:
        print("\n---------%s-----------"%cur)
        data = df_idx[[cur]]
        normalize(data, order=1)
        plt.plot(data)
        plt.xlabel("Years")
        plt.ylabel("USD/%s Prices"%cur)
        plt.grid(True)
        plt.show()
        
        diff = data.index.values[-1] - data.index.values[0]
        days = diff.astype('timedelta64[D]')
        days = days / np.timedelta64(1, 'D')
        years = abs(int(days/365))
        print(years)
        print("Total data: %d years"%years)
        print("80 percent data = 1980 to %d"%(1980 + int(0.8*years)))
        
        split_date = pd.Timestamp('01-01-2011')
        train = data.loc[:split_date]
        test = data.loc[split_date:]
        
        ax = train.plot()
        test.plot(ax=ax)
        plt.xlabel('No. of Days')
        plt.ylabel('USD/%s Prices'%cur)
        plt.xlabel("Years")
        plt.legend(['Test','Train'])
        plt.grid(True)
        plt.show()

        sc = MinMaxScaler()
        train_sc = sc.fit_transform(train)
        test_sc = sc.transform(test)

        train_sc_df = pd.DataFrame(train_sc, columns=['Y'], index=train.index)
        test_sc_df = pd.DataFrame(test_sc, columns=['Y'], index=test.index)
        for s in range(1,2):
            train_sc_df['X_{}'.format(s)] = train_sc_df['Y'].shift(s)
            test_sc_df['X_{}'.format(s)] = test_sc_df['Y'].shift(s)

        X_train = train_sc_df.dropna().drop('Y', axis=1)
        y_train = train_sc_df.dropna().drop('X_1', axis=1)

        X_test = test_sc_df.dropna().drop('Y', axis=1)
        y_test = test_sc_df.dropna().drop('X_1', axis=1)

        X_train = X_train.as_matrix()
        y_train = y_train.as_matrix()
    
        X_test = X_test.as_matrix()
        y_test = y_test.as_matrix()
        
        y_true=sc.inverse_transform(y_test)

        train_test_split = int(np.ceil(2*len(df)/float(3)))
        print('Train size: (%d x %d)'%(X_train.shape[0], X_train.shape[1]))
        print('Test size: (%d x %d)'%(X_test.shape[0], X_test.shape[1]))
        X_tr_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
##########SVR#################

        sc = MinMaxScaler()
        train_sc = sc.fit_transform(train)
        test_sc = sc.transform(test)

        train_sc_df = pd.DataFrame(train_sc, columns=['Y'], index=train.index)
        test_sc_df = pd.DataFrame(test_sc, columns=['Y'], index=test.index)
        for s in range(1,2):
            train_sc_df['X_{}'.format(s)] = train_sc_df['Y'].shift(s)
            test_sc_df['X_{}'.format(s)] = test_sc_df['Y'].shift(s)
    
        X_train = train_sc_df.dropna().drop('Y', axis=1)
        y_train = train_sc_df.dropna().drop('X_1', axis=1)

        X_test = test_sc_df.dropna().drop('Y', axis=1)
        y_test = test_sc_df.dropna().drop('X_1', axis=1)

        X_train = X_train.as_matrix()
        y_train = y_train.as_matrix()

        X_test = X_test.as_matrix()
        y_test = y_test.as_matrix()

        print('Train size: (%d x %d)'%(X_train.shape[0], X_train.shape[1]))
        print('Test size: (%d x %d)'%(X_test.shape[0], X_test.shape[1]))

        regressor = SVR(kernel='rbf', C=1e3, gamma='scale')
        regressor.fit(X_train, y_train)
        y_pred_svr = regressor.predict(X_test)


        r2_test = r2_score(y_test, y_pred_svr)
        r2_test_adj = adj_r2_score(r2_test, y_test.shape[0], y_test.shape[1])
        print("\nR-squared is: %f"%r2_test)
        print("Adjusted R-squared is: %f\n "%r2_test_adj)

###########ANN#################

        K.clear_session()
        model_ann = Sequential()
        model_ann.add(Dense(12, input_dim=1, activation='relu', kernel_initializer='lecun_uniform'))
        model_ann.add(Dense(1))
        model_ann.compile(optimizer='adam', loss='mean_squared_error')
        early_stop = EarlyStopping(monitor='loss', patience=2, verbose=2)
        history = model_ann.fit(X_train, y_train, epochs=100, batch_size=8, verbose=2, callbacks=[early_stop], shuffle=False)

        y_pred_test_ann = model_ann.predict(X_test)
        y_train_pred_ann = model_ann.predict(X_train)

        print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_ann)))
        r2_train = r2_score(y_train, y_train_pred_ann)
        print("The Adjusted R2 score on the Train set is:\t{:0.3f}".format(adj_r2_score(r2_train, X_train.shape[0], X_train.shape[1])))
        print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_ann)))
        r2_test = r2_score(y_test, y_pred_test_ann)
        print("The Adjusted R2 score on the Test set is:\t{:0.3f}\n".format(adj_r2_score(r2_test, X_test.shape[0], X_test.shape[1])))
 
########### LSTM ##################################
        
        K.clear_session()
        model_lstm = Sequential()
        model_lstm.add(LSTM(7, input_shape=(1, X_train.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
        model_lstm.add(Dense(1))
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        
        history_model_lstm = model_lstm.fit(X_tr_t, y_train, epochs=100, batch_size=16, verbose=2, shuffle=False, callbacks=[early_stop])
    
        y_pred_test_lstm = model_lstm.predict(X_tst_t)
        y_train_pred_lstm = model_lstm.predict(X_tr_t)
        
        print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
        r2_train = r2_score(y_train, y_train_pred_lstm)
        print("The Adjusted R2 score on the Train set is:\t{:0.3f}".format(adj_r2_score(r2_train, X_train.shape[0], X_train.shape[1])))
        print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
        r2_test = r2_score(y_test, y_pred_test_lstm)
        print("The Adjusted R2 score on the Test set is:\t{:0.3f}\n".format(adj_r2_score(r2_test, X_test.shape[0], X_test.shape[1])))

##########Multilayer Perceptron#########

        K.clear_session()
        model_nn = Sequential()
        model_nn.add(Dense(1, input_shape=(X_test.shape[1],), activation='tanh', kernel_initializer='lecun_uniform'))
        model_nn.compile(optimizer='adam', loss='mean_squared_error')
        model_nn.fit(X_train, y_train, batch_size=8, epochs=100, verbose=2, callbacks=[early_stop])
        
        y_pred_test_nn = model_nn.predict(X_test)
        y_train_pred_nn = model_nn.predict(X_train)
        
        print('R-Squared: %f\n'%(r2_score(y_test, y_pred_nn1)))
        print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_nn)))
        r2_train = r2_score(y_train, y_train_pred_nn)
        print("The Adjusted R2 score on the Train set is:\t{:0.3f}".format(adj_r2_score(r2_train, X_train.shape[0], X_train.shape[1])))
        print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_nn)))
        r2_test = r2_score(y_test, y_pred_test_nn)
        print("The Adjusted R2 score on the Test set is:\t{:0.3f}\n".format(adj_r2_score(r2_test, X_test.shape[0], X_test.shape[1])))
  
        K.clear_session()
        model_mlp = Sequential()
        model_mlp.add(Dense(50, input_shape=(X_test.shape[1],), activation='relu', kernel_initializer='lecun_uniform'))
        model_mlp.add(Dense(50, input_shape=(X_test.shape[1],), activation='relu'))
        model_mlp.add(Dense(1))
        model_mlp.compile(optimizer='adam', loss='mean_squared_error')
        model_mlp.fit(X_train, y_train, batch_size=8, epochs=100, verbose=2, callbacks=[early_stop])

        y_pred_test_mlp = model_mlp.predict(X_test)
        y_train_pred_mlp = model_mlp.predict(X_train)
        
        print('R-Squared: %f\n'%(r2_score(y_test, y_pred_nn2)))
        print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_mlp)))
        r2_train = r2_score(y_train, y_train_pred_mlp)
        print("The Adjusted R2 score on the Train set is:\t{:0.3f}".format(adj_r2_score(r2_train, X_train.shape[0], X_train.shape[1])))
        print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_mlp)))
        r2_test = r2_score(y_test, y_pred_test_mlp)
        print("The Adjusted R2 score on the Test set is:\t{:0.3f}\n".format(adj_r2_score(r2_test, X_test.shape[0], X_test.shape[1])))

        score_ann= model_ann.evaluate(X_test, y_test,
        batch_size=16,verbose=2) score_lstm=
        model_lstm.evaluate(X_tst_t, y_test, batch_size=16,verbose=2)
        score_nn= model_nn.evaluate(X_test, y_test, batch_size=16,verbose=2)
        score_mlp= model_mlp.evaluate(X_test, y_test, batch_size=16,verbose=2)
      
        print('ANN: %f'%score_ann)
        print('LSTM: %f'%score_lstm)
        print('NN: %f'%score_nn)
        print('MLP: %f'%score_mlp)

        
###########Graphs####################
        plt.figure(facecolor='w', edgecolor='k')
        plt.xlabel('No. of Days')
        plt.ylabel('USD/%s scaled'%cur)
        plt.plot(y_test)
        plt.plot(y_pred_svr)
        plt.plot(y_pred_test_ann)
        plt.plot(y_pred_test_lstm)
        plt.plot(y_pred_nn)
        plt.plot(y_pred_mlp)
        plt.legend(['Actual Price','SVR Prediction','ANN Prediction','LSTM Prediction','NN Prediction','MLP Prediction'])
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    df = pd.read_csv('Historical Data.csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11],parse_dates=['Date'])
    num_cores = mp.cpu_count()
    pool = Pool(os.cpu_count())
    print("No. of Cores : %d \n"%num_cores)
    mcp(df)                        
    result = pool.map(mcp(df), df)
    pool.close()
    pool.join()
    pool.clear()
