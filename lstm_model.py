from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation,Flatten,Dropout
from keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM

foxconndf= pd.read_csv('./stock_dataset2.csv', index_col=0 )
foxconndf.dropna(how='any',inplace=True)

test =pd.read_csv('./test2.csv',index_col = 0)
test.dropna(how='any',inplace=True)


def normalize(df):
    newdf= df.copy()
    min_max_scaler = preprocessing.MinMaxScaler()
    
    newdf['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))  #正規化0-1
    newdf['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    newdf['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    newdf['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))
    newdf['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))
    return newdf

def data_helper(df, time_frame,test):
    # 資料維度: 開盤價、收盤價、最高價、最低價、成交量, 5維
    number_features = len(df.columns)
    # 將dataframe 轉成 numpy array
    datavalue = df.as_matrix()
    result = []
    # 若想要觀察的 time_frame 為20天, 需要多加一天做為驗證答案
    if test == 1:
        for index in range(len(datavalue) - (time_frame+1)): # 從 datavalue 的第0個跑到倒數第 time_frame+1 個
            result.append(datavalue[index: index + (time_frame+1)]) # 逐筆取出 time_frame+1 個K棒數值做為一筆 instance
        result = np.array(result)
        number_train = round(0.9 * result.shape[0]) # 取 result 的前90% instance做為訓練資料      
        x_train = result[:int(number_train), :-1]   # 訓練資料中, 只取每一個 time_frame 中除了最後一筆的所有資料做為feature
        y_train = result[:int(number_train), -1][:,-1]  # 訓練資料中, 取每一個 time_frame 中最後一筆資料的最後一個數值(收盤價)做為答案
        
        # 測試資料
        x_test = result[int(number_train):, :-1]
        y_test = result[int(number_train):, -1][:,-1]
        
        # 將資料組成變好看一點
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], number_features))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], number_features))
        return [x_train, y_train, x_test, y_test]
    
    else:
        for index in range(int(len(datavalue)/time_frame)): # 從 datavalue 的第0個跑到倒數第 time_frame+1 個
            result.append(datavalue[index: index + (time_frame)])
        result = np.array(result)
        return result

#建立模型
def build_model(input_length, input_dim):
    d = 0.3
    model = Sequential()
    model.add(LSTM(256, input_shape=(input_length, input_dim), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(256, input_shape=(input_length, input_dim), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(16,kernel_initializer="uniform",activation='relu'))
    model.add(Dense(1,kernel_initializer="uniform",activation='linear'))
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    return model

def denormalize(df, norm_value):
    original_value = df['close'].values.reshape(-1,1)
    norm_value = norm_value.reshape(-1,1)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm_value = min_max_scaler.inverse_transform(norm_value)
    
    return denorm_value


foxconndf_norm= normalize(foxconndf)

test2 = normalize(test)

test_t = data_helper(test2,20,0)
print("test_t" , test_t.shape)

# 以20天為一區間進行股價預測
X_train, y_train, X_test, y_test = data_helper(foxconndf_norm, 20,1)

print("X_train" , X_train.shape)
print("y_train" , y_train.shape)
print("X_test" , X_test.shape)
print("y_test" , y_test.shape)

#model = build_model()
## 一個batch有128個instance，總共跑50個迭代
#model.fit(X_train, y_train, batch_size=128, epochs=50, validation_split=0.1, verbose=1)
#model.save("testmodel.h5")

#load LSTM model
model = load_model("model_256_300.h5")

print("model" , model)

# 用訓練好的 LSTM 模型對測試資料集進行預測
pred = model.predict(X_test)

# 將預測值與正確答案還原回原來的區間值
denorm_pred = denormalize(foxconndf, pred)
denorm_ytest = denormalize(foxconndf, y_test)

pred_six = []
pred_six.append(denorm_pred[-1])
six = model.predict(test_t)
denorm_pred_six = denormalize(test, six)
for i in range(len(denorm_pred_six)):
    pred_six.append(denorm_pred_six[i])

pred_six = np.array(pred_six)

print("pred_six",len(denorm_pred))
print("denorm_ytest",denorm_ytest.size)

plt.figure()
plt.plot(list(range(len(denorm_pred))),denorm_pred,color='red', label='Prediction')
plt.plot(list(range(len(denorm_ytest))),denorm_ytest,color='blue', label='Original')
plt.plot(list(range(len(denorm_ytest),len(denorm_ytest)+len(pred_six))),pred_six,color='black', label='Test')
plt.legend(loc='best')
plt.show()

close = foxconndf['close']
pred = []
pred.append(close[-1])
for i in range(len(denorm_pred)):
    pred.append(denorm_pred[i])

plt.figure()
plt.plot(list(range(len(close))),close,color='blue', label='close')
plt.plot(list(range(len(close),len(pred)+len(close))),pred,color='red', label='Prediction')
plt.plot(list(range(len(pred)+len(close),len(pred)+len(close)+len(pred_six))),pred_six,color='black', label='Test')
plt.legend(loc='best')
plt.show()

#結構
print(model.summary())

#正確率
a=denorm_pred.round(0)  #取到整數
b=denorm_ytest.round(0)
accuracy = np.mean(a == b)  #如果相等的話就是True=1
print("Prediction Accuracy: %.2f%%" % (accuracy*100))