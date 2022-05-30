from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import initializers
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.framework import tensor_shape

import Dense

class layer(Layer):
    
    def __init__(self, output_dim, backward = False, time_major = False, initializer = 'glorot_uniform', trainable = True, dynamic = True):

        super(layer, self).__init__()

        self.output_dim = output_dim
        
        self.attention_dim = output_dim ################
        
        self.backward = backward
        
        self.time_major = time_major
        self.initializer = initializers.get(initializer)
        
    
    def build(self, input_shape):
        
        self.input_dim = tensor_shape.dimension_value(input_shape[-1][-1])
        #self.input_dim = tensor_shape.dimension_value(input_shape[-1])
        #print('input_dim', self.input_dim)

        # attention weights
        self.Wa = self.add_weight(shape = (self.input_dim, self.attention_dim), initializer = self.initializer,
                                  dtype = 'float32', name = 'Wa')
        self.Ua = self.add_weight(shape = (self.input_dim, self.attention_dim), initializer = self.initializer,
                                  dtype= 'float32', name = 'Ua')
        self.v = self.add_weight(shape = (self.attention_dim,), initializer = self.initializer,
                                  dtype = 'float32', name = 'va')
        
        
        self.Wi = self.add_weight(shape = (self.output_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Wi')
        self.Ui = self.add_weight(shape = (self.output_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Ui')
        self.Ci = self.add_weight(shape = (self.output_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Ci')
        self.bi = self.add_weight(shape = (self.output_dim),
                                          initializer = initializers.get('zeros'), dtype = 'float32', name = 'bi')

        self.Wf = self.add_weight(shape = (self.output_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Wf')
        self.Uf = self.add_weight(shape = (self.output_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Uf')
        self.Cf = self.add_weight(shape = (self.output_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Cf')
        self.bf = self.add_weight(shape = (self.output_dim),
                                          initializer = initializers.get('ones'), dtype = 'float32', name = 'bf')

        self.Wc = self.add_weight(shape = (self.output_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Wc')
        self.Uc = self.add_weight(shape = (self.output_dim, self.output_dim), 
                                          initializer = self.initializer, dtype = 'float32', name = 'Uc')
        self.Cc = self.add_weight(shape = (self.output_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Cc')
        self.bc = self.add_weight(shape = (self.output_dim),
                                          initializer = initializers.get('zeros'), dtype = 'float32', name = 'bc')

        self.Wo = self.add_weight(shape = (self.output_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Wo')
        self.Uo = self.add_weight(shape = (self.output_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Uo')
        self.Co = self.add_weight(shape = (self.output_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Co')
        self.bo = self.add_weight(shape = (self.output_dim),
                                          initializer = initializers.get('zeros'), dtype = 'float32', name = 'bo')


    def step(self, cell_blocks, cell_inputs):
        
        #print(cell_blocks.shape, cell_inputs[0].shape, cell_inputs[1].shape)
        
        previous_hidden_state, previous_cell_state = array_ops.unstack(cell_blocks)
        
        current_states, h = cell_inputs
        
        #print('s', current_states.shape, 'h', h.shape)
        
        # Bahdanau attention
        A = math_ops.matmul(self.H, self.Ua)
        #print('A', A.shape)
        B = math_ops.matmul(previous_hidden_state, self.Wa)
        
        A = array_ops.transpose(A, [1, 0, 2])
        #print('A', A.shape)
        AB = A + B
        AB = array_ops.transpose(AB, [1, 0, 2])
        #print('AB', AB.shape)
        
        #print('B', B.shape)
        #print(math_ops.tanh(A + B).shape, 'v', self.v.shape)

        score = special_math_ops.einsum('bsd, d -> bs', math_ops.tanh(AB), self.v)
        #print('score', score.shape)

        a = nn_ops.softmax(score)
        #print('a', a.shape)

        #print('H', self.H.shape)
        context = special_math_ops.einsum('bsd, bs -> bd', self.H, a)
        
        #print('state', current_states.shape, previous_hidden_state.shape)
        #print('context', context.shape, self.Cf.shape, self.bf.shape)        
        


        input_gate = math_ops.sigmoid(math_ops.matmul(current_states, self.Wi) + math_ops.matmul(previous_hidden_state, self.Ui) 
                                      + math_ops.matmul(context, self.Ci) + self.bi)
        
        forget_gate = math_ops.sigmoid(math_ops.matmul(current_states, self.Wf) + math_ops.matmul(previous_hidden_state, self.Uf)
                                       + math_ops.matmul(context, self.Cf) + self.bf)

        gate_weight = math_ops.tanh(math_ops.matmul(current_states, self.Wc) + math_ops.matmul(previous_hidden_state, self.Uc)
                                    + math_ops.matmul(context, self.Cc) + self.bc)

        current_cell_state = forget_gate * previous_cell_state + input_gate * gate_weight
        
        output_gate = math_ops.sigmoid(math_ops.matmul(current_states, self.Wo) + math_ops.matmul(previous_hidden_state, self.Uo)
                                       + math_ops.matmul(context, self.Co) + self.bo)
        
        current_hidden_state = output_gate * math_ops.tanh(current_cell_state)
    
        return array_ops.stack([current_hidden_state, current_cell_state])
    

    def call(self, input_blocks, init_states):
        

        _, self.H = input_blocks

        #print(self.H.shape)
        
        #print(input_blocks, input_blocks[0].shape, input_blocks[1].shape)
        input_blocks = (array_ops.transpose(input_blocks[0], [1, 0, 2]), array_ops.transpose(input_blocks[1], [1, 0, 2]))
        #input_blocks = array_ops.transpose(input_blocks, [2, 0, 1, 3])

        #print('input_blocks', input_blocks.shape, 'H', self.H.shape)
        
        outputs = functional_ops.scan(self.step, input_blocks, initializer = init_states)
        
        outputs = array_ops.transpose(outputs, [1, 2, 0, 3])[0]

        #print('outputs', outputs.shape)
        
        return outputs

