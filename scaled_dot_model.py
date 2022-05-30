import tensorflow as tf

import Dense
import LSTM
import Attention
import bias


class construct(tf.keras.Model):
    
    def __init__(self, in_dim, out_dim, d_model, num_layers, backward = True):

        super(construct, self).__init__()

        self.num_layers = num_layers
        self.backward = backward
        
        # encoder
        self.input_dense = Dense.layer(d_model)
        self.encoder_rnn = LSTM.layer(d_model, False)

        # decoder
        self.output_dense = Dense.layer(d_model)
        self.inout_attention = Attention.layer(d_model)
        self.output_layernorm = tf.keras.layers.LayerNormalization(axis = -1, epsilon = 1e-8)
        self.decode_rnn = LSTM.layer(d_model, False)

        self.predict = Dense.layer(out_dim)
        self.bias = bias.layer(out_dim)

    def call(self, inputs, outputs, init_states):

        en_inputs = self.input_dense(inputs)        
        en_outputs = self.encoder_rnn(en_inputs, init_states)

        de_inputs = self.output_dense(outputs)        
        de_outputs = self.inout_attention(de_inputs, en_outputs, en_outputs)
        de_outputs = self.output_layernorm((de_inputs + de_outputs))
        de_outputs = self.decode_rnn(de_outputs, init_states)
        
        # outputs
        outputs = self.predict(de_outputs)
        outputs2 = self.bias(outputs)

        return outputs2



