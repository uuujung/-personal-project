from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

import Dense


class layer(Layer):
    
    def __init__(self, d_model, trainable = True, dynamic = True):

        super(layer, self).__init__()

        self.wq = Dense.layer(d_model)
        self.wk = Dense.layer(d_model)
        self.wv = Dense.layer(d_model)
        
    
    def scaled_dot_product_attention(self, q, k, v):

        qk = math_ops.matmul(q, k, transpose_b = True)
        
        dk = math_ops.cast(k.shape[-1], 'float32')
        
        scaled_attention_logits = qk #/ math_ops.sqrt(dk)
        
        attention_weights = nn_ops.softmax(scaled_attention_logits, axis = -1)
        
        out = math_ops.matmul(attention_weights, v)
        
        return out
    
    
    def call(self, q, k, v):
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
    
        outputs = self.scaled_dot_product_attention(q, k, v)
        
        return outputs