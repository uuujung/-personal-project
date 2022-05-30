""" 2021/12/02 """
import time
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from tensorflow.keras import Model

import tensorflow as tf

# 모델 선택
# import scaled_dot_model as SDAttention
# import bahdanau_model as SDAttention
# import bahdanau_selfattention_model as SDAttention
import scaled_dot_selfattention_model as SDAttention

tf.random.set_seed(3172021)


os.environ['TF.MANAGED_FORCE_DEVICE_ALLOC'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_CACHE_DISABLE'] = '0'


gpu = 0
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
tf.config.experimental.set_memory_growth(gpus[gpu], True)


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


#concat
raw_sdn = pd.concat([SP_, dow_, nas_], axis = 1)
raw_sdn.columns = ['SP', 'dow', 'nas']
raw_sdn.index = date

base_sdn = raw_sdn.iloc[0, :].values

raw_sdn /= base_sdn

raw_osg = pd.concat([oil_, silver_, gold_], axis = 1)
raw_osg.columns = ['oil', 'silver', 'gold']
raw_osg.index = date



raw_sdn_osg = pd.concat([SP_, dow_, nas_, oil_, silver_, gold_], axis = 1)
raw_sdn_osg.columns = ['SP', 'dow', 'nas', 'oil', 'silver', 'gold']
raw_sdn_osg.index = date

base_osg = raw_osg.iloc[0, :].values

raw_osg /= base_osg

raw_sdn_osg = pd.concat([SP_, dow_, nas_, oil_, silver_, gold_], axis = 1)
raw_sdn_osg.columns = ['SP', 'dow', 'nas', 'oil', 'silver', 'gold']
raw_sdn_osg.index = date

base_sdn_osg = raw_sdn_osg.iloc[0, :].values

raw_sdn_osg /= base_sdn_osg


""" prameters : for dataset """

WARM_UP_SIZE = 10
PRED_SIZE = 60 + WARM_UP_SIZE
AHEAD = 5 #
FOR_DECODER = 1

LEARN_SIZE = 240 # year
MINI_BATCH_RANGE = 60 # month
ROLL_SIZE = 1

BATCH_SIZE = 8

input_sdn = raw_sdn.iloc[-(LEARN_SIZE + FOR_DECODER + PRED_SIZE + AHEAD * 2 ):-(AHEAD), :]
output_sdn = raw_sdn.iloc[-(LEARN_SIZE + FOR_DECODER + PRED_SIZE + AHEAD):, :]

input_osg = raw_osg.iloc[-(LEARN_SIZE + FOR_DECODER + PRED_SIZE + AHEAD * 2 ):-(AHEAD), :]
output_osg = raw_osg.iloc[-(LEARN_SIZE + FOR_DECODER + PRED_SIZE + AHEAD ):, :]

input_sdn_osg = raw_sdn_osg.iloc[-(LEARN_SIZE + FOR_DECODER + PRED_SIZE + AHEAD ):-(AHEAD), :]


# normalize with learning range minimum
# input_sdn = input_sdn / input_sdn.iloc[:LEARN_SIZE].describe().transpose()['min']
# output_sdn = output_sdn / output_sdn.iloc[:LEARN_SIZE].describe().transpose()['min']

# input_osg = input_osg / input_osg.iloc[:LEARN_SIZE].describe().transpose()['min']
# output_osg = output_osg / output_osg.iloc[:LEARN_SIZE].describe().transpose()['min']

# input_sdn_osg = input_sdn_osg / input_sdn_osg.iloc[:LEARN_SIZE].describe().transpose()['min']

# input_sdn_osg.corr()


fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (12, 6))
input_sdn.plot(ax = axes[0])
output_osg.plot(ax = axes[1])


NUM_INPUT_FEATURES = input_sdn.shape[-1]
NUM_OUTPUT_FEATURES = output_osg.shape[-1]

DEPTH = NUM_INPUT_FEATURES * 4
PATIENCE = 3

MAX_EPOCHS = 10000

TOTAL_WINDOWS = (LEARN_SIZE - (MINI_BATCH_RANGE - ROLL_SIZE)) / ROLL_SIZE 

TRAIN_TAKE = TOTAL_WINDOWS // BATCH_SIZE

#==================================

in_dim = NUM_INPUT_FEATURES
out_dim = NUM_OUTPUT_FEATURES
d_model = DEPTH
num_layers = 0

model = SDAttention.construct(in_dim, out_dim, d_model, num_layers)

init_states = tf.convert_to_tensor(np.zeros([2, BATCH_SIZE, d_model], dtype = 'float32'), dtype = tf.float32)
pinit_states = tf.convert_to_tensor(np.zeros([2, 1, d_model], dtype = 'float32'), dtype = tf.float32)

optimizer = tf.optimizers.Adam(1E-3)

restore = './model_scaled_p/test_model'
restore_p = './model_scaled_p_p/test_model'


# restore = './model_bahdanau/test_model'
# restore_p = './model_bahdanau/test_model'


PREDICTIONS = list()
MAPE = list()
be_loss = list()
TRUE = list()

def window(x, y, width, step = 1, axis = 0):
    
    assert len(x) == len(y)
    
    x_ = tf.signal.frame(x, width, step, axis = axis)
    y_ = tf.signal.frame(y, width, step, axis = axis)
    
    return x_, y_


for day in range(PRED_SIZE):

    start_time = time.time()
    
    #train input, output
    inputs_ = input_osg[(-PRED_SIZE + day - (LEARN_SIZE + AHEAD + FOR_DECODER)):(-PRED_SIZE + day - (AHEAD + FOR_DECODER)+1)]
    outputs_ = output_sdn[(-PRED_SIZE + day - (LEARN_SIZE + AHEAD + FOR_DECODER) ):(-PRED_SIZE + day - (AHEAD + FOR_DECODER) +1 )]
    
    
    #차분===========================
    inputs = inputs_.diff().dropna()
    outputs = outputs_.diff().dropna()
    #==============================
    
    # min normal ================
    # normal_min_in = inputs.describe().transpose()['min']
    # inputs = inputs / inputs.describe().transpose()['min']
    # normal_min_out = outputs.describe().transpose()['min']
    # outputs = outputs / outputs.describe().transpose()['min']

    # if day == (PRED_SIZE-1) :
    #     pred_inputs = input_osg[-5:] / normal_min_in
    #     pred_outputs = output_sdn[-10:] / normal_min_out
    # else:
    #     pred_inputs = input_osg[(-PRED_SIZE + day - AHEAD +1):(-PRED_SIZE + day +1)] / normal_min_in
    #     pred_outputs = output_sdn[(-PRED_SIZE + day - (AHEAD * 2) +1):(-PRED_SIZE + day +1)] / normal_min_out
    #============================================
    
    #pred input, output
    if day == (PRED_SIZE-1) :
        pred_inputs_ = input_osg[-6:]
        pred_outputs_ = output_sdn[-11:]
        
        pred_inputs = pred_inputs_.diff().dropna()
        pred_outputs =  pred_outputs_.diff().dropna()
    else:
        pred_inputs_ = input_osg[(-PRED_SIZE + day - AHEAD):(-PRED_SIZE + day +1)]
        pred_outputs_ = output_sdn[(-PRED_SIZE + day - (AHEAD * 2)):(-PRED_SIZE + day +1)]
        
        pred_inputs = pred_inputs_.diff().dropna()
        pred_outputs =  pred_outputs_.diff().dropna()
           
       
    inputs = inputs.astype('float32')
    outputs = outputs.astype('float32')
    pred_inputs = pred_inputs.astype('float32')
    pred_outputs = pred_outputs.astype('float32')
    
    
    xy = window(inputs, outputs, MINI_BATCH_RANGE)
    #print(len(xy[0]))
    train_ds = tf.data.Dataset.from_tensor_slices(xy)
    
    train_ds = train_ds.shuffle(len(xy[0])).batch(BATCH_SIZE, drop_remainder = True
                                                  ).prefetch(BATCH_SIZE * 4).repeat()
    
    learning_rate = 1E-3
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    if day > 0:
        print('\n\tLoad weights...', end = '')
        model.load_weights(restore)
        print('Done\n')

    previous_loss = 0.
    previous_mae = 0.
    best_loss = 1E+5
    best_mae = 1.
    epoch_mae = 0.
    epoch_loss = 0.
    early_stop_step = 0
    epoch = 1#int(MAX_EPOCHS)

    for iepoch, train_data in enumerate(train_ds.take(int(TRAIN_TAKE * MAX_EPOCHS))):
    
        inputs, outputs = train_data
   
        with tf.GradientTape() as tape:
                
            predict = model(inputs, outputs, init_states)
            
            loss = tf.reduce_mean(100. * tf.abs(predict - outputs ) / (outputs + 1E-5))
            
            # loss = tf.reduce_mean(tf.abs(predict - outputs) ** 2)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        epoch_loss += loss

        # #pred
        # predicted = model(pred_inputs.values[np.newaxis, :, :], pred_outputs.values[np.newaxis, :-5, :] , pinit_states)
        # predicted = np.squeeze(predicted)[-1, :]
        
        #test_mae = tf.reduce_mean(np.abs(pred_outputs.values[-1, :] - predicted))

        epoch_mae += tf.reduce_mean(tf.abs(predict - outputs))
        
        #print('iepoch %d: loss = %.4f' % (iepoch + 1, loss.numpy()))

        if (iepoch + 1) % int(TRAIN_TAKE) == 0:
            
            epoch_loss /= int(TRAIN_TAKE)
            epoch_mae /= int(TRAIN_TAKE)
            
            if best_mae < epoch_mae:    # test mae monitor for early stopp....
                early_stop_step += 1

                if early_stop_step > PATIENCE:
                    print('\t\t===> Early stopped at epoch %d: loss %.4f(%.4f) took %.4f' % (epoch, epoch_loss, best_loss, time.time() - start_time))
                    model.load_weights(restore)
                    be_loss.append(best_loss)
                    break
            else:
                early_stop_step = 0
                best_loss = epoch_loss
                best_mae = epoch_mae
                model.save_weights(restore)

            crit = abs(previous_mae - epoch_mae) 
            print('\t\t===> Epoch %d: loss %.4f(%.4f),  MAE : %.4f(%.4f)' % (epoch, epoch_loss, best_loss, epoch_mae, best_mae))
            
            
            if crit < 1E-1 * previous_mae:
                
                print('\n\t\t===> Converged at epoch %d' % epoch)
                be_loss.append(best_loss)
                break
            
            
            previous_loss = epoch_loss
            previous_mae = epoch_mae
            epoch_loss = 0.
            epoch_mae = 0.
            epoch += 1
            

 ##############################################################################           

    # pred == 위치 변경
    predicted = model(pred_inputs.values[np.newaxis, :, :], pred_outputs.values[np.newaxis, :-5, :] , pinit_states)

    predicted = np.squeeze(predicted)[-1, :]
    
    #원스케일
    predicted = np.array(pred_outputs_[-2:-1]) + predicted
    # ===
    
    PREDICTIONS.append(predicted)
    #PREDICTIONS.append(predicted[-1])
    
    
    print(pred_outputs_.values[-1, :])    
    print(predicted)
    mape = np.abs(pred_outputs_.values[-1, :] - predicted) / pred_outputs_.values[-1, :]
    TRUE.append(pred_outputs_.values[-1, :])
    
    print(mape)
    
    MAPE.append(mape)
    print('day', day + 1 ,"Done")
    

PREDICTIONS2 = PREDICTIONS[WARM_UP_SIZE:]
MAPE = np.asarray(MAPE[WARM_UP_SIZE:])    
TRUE = TRUE[WARM_UP_SIZE:]

# print(output_osg.columns)
# print(MAPE)
# print(1. - np.mean(MAPE, axis = 0))


# plot========================================================================

# pred_df = pd.DataFrame(PREDICTIONS)
# pred_df.columns = ['pred_oil', 'pred_silver', 'pred_gold']
# pred_df.index = output_sdn.index[-(PRED_SIZE) + WARM_UP_SIZE:]


# plot_data = pd.merge(input_sdn, output_osg, on = 'Date', how = 'outer') 
# #how : outer(합집합) inner (교집합) 
# # on :merge의 기준이 되는 Key 변수


# plot_data = pd.merge(plot_data, pred_df, on = 'Date', how = 'outer')

# fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (15, 12), constrained_layout=True)

# ax1 = plot_data['oil'].plot(ax = axes[0])
# ax1.title.set_text("oil(%f)" % (1. - np.mean(MAPE, axis = 0))[0])
# ax1.set_xticklabels(plot_data.index, rotation = 45) # x축 글자 라벨
# ax2 = plot_data['pred_oil'].plot(ax = axes[0])
# ax2.legend()


# ax1 = plot_data['silver'].plot(ax = axes[1])
# ax1.title.set_text("silver(%f)" % (1. - np.mean(MAPE, axis = 0))[1])
# ax1.set_xticklabels(plot_data.index, rotation = 45)
# ax2 = plot_data['pred_silver'].plot(ax = axes[1])
# ax2.legend()

# ax1 = plot_data['gold'].plot(ax = axes[2])
# ax1.title.set_text("gold(%f)" % (1. - np.mean(MAPE, axis = 0))[2])
# ax1.set_xticklabels(plot_data.index, rotation = 45)
# ax2 = plot_data['pred_gold'].plot(ax = axes[2])
# ax2.legend()


#########SP2019


pred_df = pd.DataFrame(np.squeeze(PREDICTIONS2))
pred_df.columns = ['pred_SP', 'pred_dow','pred_nas']
#pred_df = pred_df[['pred_SP', 'pred_dow','pred_nas']]

pred_df.index = output_osg[-60:].index
output_sdn2 = output_sdn # / output_sdn .describe().transpose()['min']

plot_data = pd.merge(output_sdn2, pred_df, on = 'Date', how = 'outer')

fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (15, 12), constrained_layout=True)

ax1 = plot_data['SP'].plot(ax = axes[0])
ax1.title.set_text("SP(%f)" % (1 - np.mean(MAPE, axis = 0))[0][0])
#ax1.set_xticklabels(plot_data.index, rotation = 45) # x축 글자 라벨
ax2 = plot_data['pred_SP'].plot(ax = axes[0])
ax2.legend()


ax1 = plot_data['dow'].plot(ax = axes[1])
ax1.title.set_text("dow(%f)" %  (1 - np.mean(MAPE, axis = 0))[0][1])
#ax1.set_xticklabels(plot_data.index, rotation = 45)
ax2 = plot_data['pred_dow'].plot(ax = axes[1])
ax2.legend()

ax1 = plot_data['nas'].plot(ax = axes[2])
ax1.title.set_text("nas(%f)" %  (1 - np.mean(MAPE, axis = 0))[0][2])
#ax1.set_xticklabels(plot_data.index, rotation = 45)
ax2 = plot_data['pred_nas'].plot(ax = axes[2])
ax2.legend()







# # train_min ====================================================
# pred_df = pd.DataFrame(np.squeeze(PREDICTIONS2))
# pred_df.columns = ['pred_SP', 'pred_dow','pred_nas']
# #pred_df = pred_df[['pred_SP', 'pred_dow','pred_nas']]

# pred_df.index = output_osg[-60:].index
# #df.index = output_osg.index
# #train_norml
# df2 = output_sdn / output_sdn.describe().transpose()['min']
# DF = df2[:-60][["SP","dow",'nas']]

# TRUE = pd.DataFrame(TRUE)
# TRUE.index = output_sdn[-60:].index
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
