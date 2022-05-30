""" 2021/12/02 """

import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt


os.environ['TF.MANAGED_FORCE_DEVICE_ALLOC'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_CACHE_DISABLE'] = '0'


from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from tensorflow.keras import Model

import tensorflow as tf


import SeriesAttention as Attention


gpu = 0
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
tf.config.experimental.set_memory_growth(gpus[gpu], True)


tf.random.set_seed(3172021)


def window(x, y, width, step = 1, axis = 0):
    
    assert len(x) == len(y)
    
    x_ = tf.signal.frame(x, width, step, axis = axis)
    y_ = tf.signal.frame(y, width, step, axis = axis)

    return x_, y_
    #return x_ / tf.math.reduce_min(x_, axis = 1, keepdims = True), y_

#test = window(input_sdn[:-30], output_osg[:-30])

""" parameters """


# DATA Processing==============================================================


#raw data 불러오기
oil = pd.read_csv('./data/OIL19.csv')
gold = pd.read_csv('./data/GOLD19.csv')
SP = pd.read_csv('./data/S&P19.csv')
silver = pd.read_csv('./data/SILVER19.csv')
nas = pd.read_csv('./data/NASDAQ19.csv')
dow = pd.read_csv('./data/DOW19.csv')

#필요한 데이터만 추출 (Adj Close 데이터만 사용)
date = SP.pop('Date')
SP_ = SP.pop('Adj Close')
nas_ = nas.pop('Adj Close')
oil_ = oil.pop('Adj Close')
gold_ = gold.pop('Adj Close')
dow_ = dow.pop("Adj Close")
silver_=silver.pop('Adj Close')


#concat, raw_sdn 만들기=====================
raw_sdn = pd.concat([SP_, dow_, nas_], axis = 1)
raw_sdn.columns = ['SP', 'dow', 'nas']
raw_sdn.index = date

base_sdn = raw_sdn.iloc[0, :].values

raw_sdn /= base_sdn #2019-01-02일 값을 기준으로 표준화
#=========================================

#raw_osg 만들기 =============================
raw_osg = pd.concat([oil_, silver_, gold_], axis = 1)
raw_osg.columns = ['oil', 'silver', 'gold']
raw_osg.index = date

base_osg = raw_osg.iloc[0, :].values

raw_osg /= base_osg #2019-01-02일 값을 기준으로 표준화
#=================================

#raw_sdn_osg 만들기 ================
raw_sdn_osg = pd.concat([SP_, dow_, nas_, oil_, silver_, gold_], axis = 1)
raw_sdn_osg.columns = ['SP', 'dow', 'nas', 'oil', 'silver', 'gold']
raw_sdn_osg.index = date

base_sdn_osg = raw_sdn_osg.iloc[0, :].values

raw_sdn_osg /= base_sdn_osg #2019-01-02일 값을 기준으로 표준화
#==============================


""" prameters : for dataset """

WARM_UP_SIZE = 0
PRED_SIZE = 60 + WARM_UP_SIZE
AHEAD = 5 #
FOR_DECODER = 1

LEARN_SIZE = 240 # year
MINI_BATCH_RANGE = 60 # month
ROLL_SIZE = 1

BATCH_SIZE = 8

input_sdn = raw_sdn.iloc[-(LEARN_SIZE  + PRED_SIZE + AHEAD * 3 ):-(AHEAD), :]
output_sdn = raw_sdn.iloc[-(LEARN_SIZE  + PRED_SIZE + AHEAD*2):, :]

input_osg = raw_osg.iloc[-(LEARN_SIZE  + PRED_SIZE + AHEAD * 3 ):-(AHEAD), :]
output_osg = raw_osg.iloc[-(LEARN_SIZE + PRED_SIZE + AHEAD*2 ):, :]

input_sdn_osg = raw_sdn_osg.iloc[-(LEARN_SIZE  + PRED_SIZE + AHEAD ):-(AHEAD), :]


#input과 output 엇갈린 날짜를 하나로 합치기 위해서. (input 날짜 기준이고 output은 +5일뒤임) ===============
input_osg.index = range(0, len(input_sdn))
output_sdn.index = range(0, len(output_osg))

df= pd.concat([input_osg, output_sdn], axis = 1)

df.index = input_sdn.index

input_sdn_osg = df
# input_sdn_osg_base  = input_sdn_osg.iloc[:LEARN_SIZE].describe().transpose()['min']


#===============
#input_osg, output_sdn의 인덱스가 바뀐걸 다시 돌려주기 위해 (그래프 그릴때 인덱스 맞추려고 필요) 
input_osg = raw_osg.iloc[-(LEARN_SIZE  + PRED_SIZE + AHEAD * 3 ):-(AHEAD), :]
output_sdn = raw_sdn.iloc[-(LEARN_SIZE  + PRED_SIZE + AHEAD*2):, :]

# output_sdn_base = output_sdn.iloc[:LEARN_SIZE].describe().transpose()['min']
#=========



sdn_true = np.asarray(output_sdn[-(PRED_SIZE):])
osg_true = np.asarray(output_osg[-(PRED_SIZE):])


# 표준화 이건 X 지워도 될것같음..======================================
# input_sdn_osg = input_sdn_osg / input_sdn_osg.iloc[1:LEARN_SIZE+1].describe().transpose()['min']
# output_sdn = output_sdn / output_sdn.iloc[1:LEARN_SIZE+1].describe().transpose()['min']

# normal
# input_sdn_osg = (input_sdn_osg - input_sdn_osg.iloc[1:LEARN_SIZE+1].describe().transpose()['mean']) / input_sdn_osg.iloc[1:LEARN_SIZE+1].describe().transpose()['std']
# output_sdn = (output_sdn - output_sdn.iloc[1:LEARN_SIZE+1].describe().transpose()['mean']) / output_sdn.iloc[1:LEARN_SIZE+1].describe().transpose()['std']

#===========================================

fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (12, 6))
input_sdn.plot(ax = axes[0])
output_osg.plot(ax = axes[1])


NUM_INPUT_FEATURES = input_sdn.shape[-1]
NUM_OUTPUT_FEATURES = output_osg.shape[-1]



# for dense
def dummy(x, y):
    
    return x[:, -1, :], y[:, -1, :]



""" paramters : for model / learn """

NUM_INPUT_FEATURES = input_sdn.shape[-1]
NUM_INPUT_FEATURES2 = input_sdn_osg.shape[-1]

NUM_OUTPUT_FEATURES = output_osg.shape[-1]

DEPTH = NUM_INPUT_FEATURES * 4

PATIENCE = 100

MAX_EPOCHS = 10000

TOTAL_WINDOWS = (LEARN_SIZE - (MINI_BATCH_RANGE - ROLL_SIZE)) / ROLL_SIZE 

TRAIN_TAKE = TOTAL_WINDOWS // BATCH_SIZE

""" build model """
# # lstm(S2S)
# inputs = Input((MINI_BATCH_RANGE, NUM_INPUT_FEATURES)) #[None, 60, 3]
# outputs_ = LSTM(NUM_INPUT_FEATURES)(inputs) #[None, 3]
# outputs_ = Dense(DEPTH, activation = 'relu')(outputs_) #[None, 12]
# predictions = Dense(NUM_OUTPUT_FEATURES)(outputs_) # [None, 3]

# #lstm
# inputs = Input((MINI_BATCH_RANGE, NUM_INPUT_FEATURES2)) #[None, 60, 6]
# outputs_ = LSTM(NUM_INPUT_FEATURES2)(inputs) #[None, 6]
# outputs_ = Dense(DEPTH, activation = 'relu')(outputs_) #[None, 12]
# predictions = Dense(NUM_OUTPUT_FEATURES)(outputs_) # [None, 3]


# attention(S2S)
# inputs = Input((MINI_BATCH_RANGE, NUM_INPUT_FEATURES)) #[None, 60, 3]
# outputs_ = Attention.layer(NUM_INPUT_FEATURES)(inputs) #[None, 3]
# outputs_ = Dense(DEPTH, activation = 'relu')(outputs_) #[None, 12]
# predictions = Dense(NUM_OUTPUT_FEATURES)(outputs_) # [None, 3]

# attention()
inputs = Input((MINI_BATCH_RANGE, NUM_INPUT_FEATURES2)) #[None, 60, 6]
outputs_ = Attention.layer(NUM_INPUT_FEATURES2)(inputs) #[None, 6]
outputs_ = Dense(DEPTH, activation = 'relu')(outputs_) #[None, 12]
predictions = Dense(NUM_OUTPUT_FEATURES)(outputs_) # [None, 3]



#ann(S2S)
# inputs = Input((NUM_INPUT_FEATURES))
# outputs_ = Dense(DEPTH, activation = 'relu')(inputs)
# outputs_ = Dense(DEPTH, activation = 'relu')(outputs_)
# outputs_ = Dense(DEPTH, activation = 'relu')(outputs_)
# outputs_ = Dense(DEPTH, activation = 'relu')(outputs_)
# predictions = Dense(NUM_OUTPUT_FEATURES)(outputs_)

#ann
# inputs = Input((NUM_INPUT_FEATURES2))
# outputs_ = Dense(DEPTH, activation = 'relu')(inputs)
# outputs_ = Dense(DEPTH, activation = 'relu')(outputs_)
# outputs_ = Dense(DEPTH, activation = 'relu')(outputs_)
# outputs_ = Dense(DEPTH, activation = 'relu')(outputs_)
# predictions = Dense(NUM_OUTPUT_FEATURES)(outputs_)




model = Model(inputs = [inputs], outputs = [predictions])

#predictions = model.predict(pred_input[np.newaxis, :, :])
#print(predictions.shape)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss',
                                                  patience = PATIENCE,
                                                  mode = 'min')

#loss = tf.losses.MeanSquaredError()
loss = tf.losses.MeanAbsolutePercentageError()

optimizer = tf.optimizers.Adam(1E-3)

#mape = tf.metrics.MeanAbsolutePercentageError()
mae = tf.metrics.MeanAbsoluteError()

model.compile(loss = loss, optimizer = optimizer, metrics = [mae])
    
def att_dummy(x, y):
    return x, y[:, -1, :]


PREDICTIONS = list()
MAPE = list()
TRUE = list()

for day in range(PRED_SIZE):
    
    #데이터 1 sdn_osg -> sdn 
    inputs = input_sdn_osg[(-PRED_SIZE + day - (LEARN_SIZE + AHEAD*2) + 1 ):(-PRED_SIZE + day - (AHEAD*2) + 1 )] 
    outputs = output_sdn[(-PRED_SIZE + day - (LEARN_SIZE + AHEAD) + 1 ):(-PRED_SIZE + day - (AHEAD )+1 )]
    
    #매 턴마다 표준화 =============================================
    # #min
    # normal_min = inputs.describe().transpose()['min']
    # inputs = inputs / inputs.describe().transpose()['min']
    # outputs = outputs / outputs.describe().transpose()['min']
    
    # #normal
    # normal_mean = inputs.describe().transpose()['mean']
    # normal_sd = inputs.describe().transpose()['std']
    # inputs = (inputs - normal_mean) / normal_sd + 1E-8
    # outputs = (outputs - outputs.describe().transpose()['mean']) / outputs.describe().transpose()['std'] + 1E-8
    # #===================
    
    # inputs, outputs = input_sdn_osg[day:(-PRED_SIZE + day)], output_osg[day:(-PRED_SIZE + day)]
    
    # 데이터 2 (for S2S) osg->sdn
    #inputs, outputs = input_osg[day:(-PRED_SIZE + day)], output_sdn[day:(-PRED_SIZE + day)]
    
    
    xy = window(inputs, outputs, MINI_BATCH_RANGE)

    train_ds = tf.data.Dataset.from_tensor_slices(xy)
    
    train_ds = train_ds.shuffle(len(xy[0])).batch(BATCH_SIZE).prefetch(BATCH_SIZE * 4).repeat()
    
    # dummy함수는 attention,lstm 같이, ann 따로 라서 여기서 설정해주면 됨 ===
    train_ds = train_ds.map(att_dummy)
    # train_ds = train_ds.map(dummy) #ann
    # ======================================================================
        
    history = model.fit(train_ds, 
                        epochs = MAX_EPOCHS, 
                        callbacks = [early_stopping], 
                        steps_per_epoch = TRAIN_TAKE) 

    if day >= WARM_UP_SIZE: 
        
        # dense : sdn -> osg : x -> y
        # dense/lstm/attention : sdn_osg -> osg / sdn-> t-10 , osg->t-5 / osg -> t , sdn-> t-5, osg -> t : t+5
        # encoder sdn -> t-6 decoder osg -> t-1 pred_input: encoder : t, decoder : t -> t+5 

        #sdn_osg -> sdn   
        pred_input = input_sdn_osg[-(MINI_BATCH_RANGE + PRED_SIZE - day )- (AHEAD-1):(-PRED_SIZE + day - (AHEAD-1))]
        #osg ->sdn
        #pred_input = input_osg[-(MINI_BATCH_RANGE + PRED_SIZE - day )- (AHEAD-1):(-PRED_SIZE + day - (AHEAD-1))]
      
        #pred 표준화======================================================
        # #normal
        # pred_input = (pred_input - pred_input.describe().transpose()['mean']) / pred_input.describe().transpose()['std'] + 1E-8
        # #min
        # pred_input = pred_input / normal_min # min
        # #====================================================================
        
        pred_input = np.asarray(pred_input) 

        predicted = model.predict(pred_input[np.newaxis, :, :])
        
        predicted = np.squeeze(predicted)
        
        #ann================
        
        #원스케일로 ==========================================
        # #min
        # predicted = predicted[-1]*normal_min[['SP',"dow",'nas']] #min 원스케일
        # #normal
        # predicted = predicted[-1]*normal_sd[['SP',"dow",'nas']] + normal_mean[['SP',"dow",'nas']]
        # ===================
        
        # predicted = np.array(predicted[-1])
        predicted = np.array(predicted)
        
        PREDICTIONS.append(predicted)
        # PREDICTIONS.append(predicted[-1])
        
        # train_min mape
        # sdn_true2 = np.squeeze(sdn_true[day:day+1]) / np.array(normal_min[['SP','dow', 'nas']])
        # TRUE.append(sdn_true2) 
        # mape = np.abs((sdn_true2 - predicted) / sdn_true2)
        # MAPE.append(mape)
        # # ==========
        
        #2019 mape
        mape = np.abs((np.squeeze(sdn_true[day:day+1])) - predicted) / (np.squeeze(sdn_true[day:day+1]))
        mape = np.array(mape)
        MAPE.append(mape)
        
        # MAPE.append(mape[-1])
        
        ###########################################################
        # predicted = model.predict(input_sdn[:(-PRED_SIZE + day)])
    
        # PREDICTIONS.append(predicted[-1])
        
        # mape = np.abs((outputs.values[-1] - predicted[-1])) / outputs.values[-1]
        
        # MAPE.append(mape)
    
        
    

#########################################################
#print(output_sdn.columns)
#print(1. - np.mean(MAPE, axis = 0))

# plotting

# pred_df = pd.DataFrame(PREDICTIONS)
# pred_df.columns = ['pred_oil', 'pred_silver', 'pred_gold']
# pred_df.index = output_sdn.index[(-PRED_SIZE + WARM_UP_SIZE):]

# plot_data = pd.merge(input_sdn, output_osg, on = 'Date', how = 'outer') 
# #how : outer(합집합) inner (교집합) 
# # on :merge의 기준이 되는 Key 변수


# plot_data = pd.merge(plot_data, pred_df, on = 'Date', how = 'outer')

# fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (15, 9), constrained_layout=True)

# ax1 = plot_data['oil'].plot(ax = axes[0])
# ax1.title.set_text("oil(%f)" % (1. - np.mean(MAPE, axis = 0))[0][0])
# ax1.set_xticklabels(plot_data.index, rotation = 45) # x축 글자 라벨
# ax2 = plot_data['pred_oil'].plot(ax = axes[0])
# ax2.legend()


# ax1 = plot_data['silver'].plot(ax = axes[1])
# ax1.title.set_text("silver(%f)" % (1. - np.mean(MAPE, axis = 0))[0][1])
# ax1.set_xticklabels(plot_data.index, rotation = 45)
# ax2 = plot_data['pred_silver'].plot(ax = axes[1])
# ax2.legend()

# ax1 = plot_data['gold'].plot(ax = axes[2])
# ax1.title.set_text("gold(%f)" % (1. - np.mean(MAPE, axis = 0))[0][2])
# ax1.set_xticklabels(plot_data.index, rotation = 45)
# ax2 = plot_data['pred_gold'].plot(ax = axes[2])
# ax2.legend()


###### sp nas dow

# pred_df = pd.DataFrame(PREDICTIONS)
# pred_df.columns = ['pred_SP', 'pred_dow', 'pred_nas']
# pred_df.index = output_sdn.index[(-PRED_SIZE + WARM_UP_SIZE):]

# plot_data = pd.merge(input_osg, output_sdn, on = 'Date', how = 'outer') 
# #how : outer(합집합) inner (교집합) 
# # on :merge의 기준이 되는 Key 변수


# plot_data = pd.merge(plot_data, pred_df, on = 'Date', how = 'outer')

# fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (15, 9), constrained_layout=True)

# ax1 = plot_data['SP'].plot(ax = axes[0])

# ax1.title.set_text("SP(%f)" % (1. - np.mean(MAPE, axis = 0))[0])
# ax1.set_xticklabels(plot_data.index, rotation = 45) # x축 글자 라벨
# ax2 = plot_data['pred_SP'].plot(ax = axes[0])
# ax2.legend()


# ax1 = plot_data['nas'].plot(ax = axes[1])
# ax1.title.set_text("nas(%f)" % (1. - np.mean(MAPE, axis = 0))[0])
# ax1.set_xticklabels(plot_data.index, rotation = 45)
# ax2 = plot_data['pred_nas'].plot(ax = axes[1])
# ax2.legend()

# ax1 = plot_data['dow'].plot(ax = axes[2])
# ax1.title.set_text("dow(%f)" % (1. - np.mean(MAPE, axis = 0))[0])
# ax1.set_xticklabels(plot_data.index, rotation = 45)
# ax2 = plot_data['pred_dow'].plot(ax = axes[2])
# ax2.legend()




############## train_min
# pred_df = pd.DataFrame(np.squeeze(PREDICTIONS))
# pred_df.columns = ['pred_SP', 'pred_dow','pred_nas']
# #pred_df = pred_df[['pred_SP', 'pred_dow','pred_nas']]

# pred_df.index = output_osg[-60:].index
# df.index = output_osg.index

# #train_norml
# #df2 = df / df.describe().transpose()['min']
# df2 = df / df.describe().transpose()['min']
# DF = df2[:-60][["SP","dow",'nas']]
# TRUE = pd.DataFrame(TRUE)
# TRUE.index = df2[-60:].index
# TRUE.columns = ['SP', 'dow','nas']

# df3 = pd.concat([DF,TRUE], axis = 0)


# plot_data = pd.merge(df3, pred_df, on = 'Date', how = 'outer')

# fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (15, 12), constrained_layout=True)

# ax1 = plot_data['SP'].plot(ax = axes[0])
# ax1.title.set_text("SP(%f)" % (1- np.mean(MAPE,axis = 0))[0])
# #ax1.set_xticklabels(plot_data.index, rotation = 45) # x축 글자 라벨
# ax2 = plot_data['pred_SP'].plot(ax = axes[0])
# ax2.legend()


# ax1 = plot_data['dow'].plot(ax = axes[1])
# ax1.title.set_text("dow(%f)" %(1- np.mean(MAPE,axis = 0))[1])
# #ax1.set_xticklabels(plot_data.index, rotation = 45)
# ax2 = plot_data['pred_dow'].plot(ax = axes[1])
# ax2.legend()

# ax1 = plot_data['nas'].plot(ax = axes[2])
# ax1.title.set_text("nas(%f)" % (1- np.mean(MAPE,axis = 0))[2])
# #ax1.set_xticklabels(plot_data.index, rotation = 45)
# ax2 = plot_data['pred_nas'].plot(ax = axes[2])
# ax2.legend()


# 2019========
pred_df = pd.DataFrame(np.squeeze(PREDICTIONS))
pred_df.columns = ['pred_SP', 'pred_dow','pred_nas']
#pred_df = pred_df[['pred_SP', 'pred_dow','pred_nas']]

pred_df.index = output_osg[-60:].index
df.index = output_osg.index

plot_data = pd.merge(df, pred_df, on = 'Date', how = 'outer')

fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (15, 12), constrained_layout=True)

ax1 = plot_data['SP'].plot(ax = axes[0])
ax1.title.set_text("SP(%f)" % (1- np.mean(MAPE,axis = 0))[0])
#ax1.set_xticklabels(plot_data.index, rotation = 45) # x축 글자 라벨
ax2 = plot_data['pred_SP'].plot(ax = axes[0])
ax2.legend()


ax1 = plot_data['dow'].plot(ax = axes[1])
ax1.title.set_text("dow(%f)" %(1- np.mean(MAPE,axis = 0))[1])
#ax1.set_xticklabels(plot_data.index, rotation = 45)
ax2 = plot_data['pred_dow'].plot(ax = axes[1])
ax2.legend()

ax1 = plot_data['nas'].plot(ax = axes[2])
ax1.title.set_text("nas(%f)" % (1- np.mean(MAPE,axis = 0))[2])
#ax1.set_xticklabels(plot_data.index, rotation = 45)
ax2 = plot_data['pred_nas'].plot(ax = axes[2])
ax2.legend()
