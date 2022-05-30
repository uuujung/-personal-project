import tensorflow as tf
#from tensorflow.python.ops import math_ops

import Dense
import LSTM
import AttentionLSTM_badanau as AttentionLSTM 
# import bias

#from tensorflow.python.keras.engine.base_layer import Layer


class construct(tf.keras.Model):
    
    def __init__(self, in_dim, out_dim, d_model, num_layers, backward = True):

        super(construct, self).__init__()

        self.num_layers = num_layers
        self.backward = backward
    
        # input encoder
        self.input_dense = Dense.layer(d_model)
        self.in_enrnn = LSTM.layer(d_model, False)

        # output - in docoder
        self.output_dense = Dense.layer(d_model)
        self.out_in_dernn = AttentionLSTM.layer(d_model, False)
        self.out_enrnn = [LSTM.layer(d_model, False) for _ in range(self.num_layers)]
       # output dense
        self.out_key_in_deodense = Dense.layer(out_dim)
        # self.bias = bias.layer(out_dim)

        

    def call(self, inputs, outputs, init_states):
        

        in_inputs = self.input_dense(inputs)
        in_enoutputs = self.in_enrnn(in_inputs, init_states) # 배치, 윈도우, d_model
 
        out_inputs = self.output_dense(outputs)
        out_enoutputs = out_inputs
        for i in range(self.num_layers):
            out_enoutputs = self.out_enrnn[i](out_enoutputs, init_states) #배치 윈도우 d_model
    
        out_in_deinputs = (out_enoutputs, in_enoutputs)
        out_in_deoutputs = self.out_in_dernn(out_in_deinputs, init_states) #enoutputs_[:, :, -1, :]) #배치,윈도우,d_model

        out_key_in_outputs_ = self.out_key_in_deodense(out_in_deoutputs) #배치, 윈도우, out dim

        #out_key_in_outputs_2 = self.bias(out_key_in_outputs_)
        
        return out_key_in_outputs_

