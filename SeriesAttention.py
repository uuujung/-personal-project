from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops



class layer(Layer):
    
    def __init__(self, d_model, inference = False, *args, **kwargs):

        super(layer, self).__init__()
        self.d_model = d_model        

        """ series attention weights """
        self.Wsa = self.add_weight('Wsa',
                                    shape = (self.d_model, self.d_model),
                                    initializer = 'glorot_uniform',
                                    dtype = 'float32',
                                    trainable = True)

        self.bsa = self.add_weight('bsa',
                                    shape = (self.d_model,),
                                    initializer = 'glorot_uniform',
                                    dtype = 'float32',
                                    trainable = True)

        """ series context """
        self.usa = self.add_weight('usa',
                                    shape = (self.d_model,),
                                    initializer = 'glorot_uniform',
                                    dtype = 'float32',
                                    trainable = True)        
        
        
    def call(self, x):
        
        """ [batch_size, period, features] """
        u = special_math_ops.einsum('bad, de -> bae', x, self.Wsa) + self.bsa
        u = math_ops.tanh(u)
        
        """ [batch_size, feature] """
        s = special_math_ops.einsum('bam, m -> ba', u, self.usa)
    
        
        """ [batch_size, feature] """
        self.ans_att = nn_ops.softmax(s)
        # self.embedded_article_a = tf.nn.softmax(s)
        
        """ [batch_size, feature] """
        # embedded_article = tf.einsum('bam, bm -> bm', x, a)
        embedded_answer = special_math_ops.einsum('bam, ba -> bm', x, self.ans_att)
        
        return embedded_answer # [batch, feature]
    
    