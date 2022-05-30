from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import initializers
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.framework import tensor_shape


class layer(Layer):
    
    def __init__(self, output_dim, backward = False, time_major = False, initializer = 'glorot_uniform', trainable = True, dynamic = True):

        super(layer, self).__init__()

        self.output_dim = output_dim
        
        self.backward = backward
        
        self.time_major = time_major
        self.initializer = initializers.get(initializer)
                
    
    def build(self, input_shape):
        
        self.input_dim = tensor_shape.dimension_value(input_shape[-1])

        self.Wi = self.add_weight(shape = (self.input_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Wi')
        self.Ui = self.add_weight(shape = (self.output_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Ui')
        self.bi = self.add_weight(shape = (self.output_dim),
                                          initializer = initializers.get('ones'), dtype = 'float32', name = 'bi')

        self.Wf = self.add_weight(shape = (self.input_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Wf')
        self.Uf = self.add_weight(shape = (self.output_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Uf')
        self.bf = self.add_weight(shape = (self.output_dim),
                                          initializer = initializers.get('ones'), dtype = 'float32', name = 'bf')

        self.Wc = self.add_weight(shape = (self.input_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Wc')
        self.Uc = self.add_weight(shape = (self.output_dim, self.output_dim), 
                                          initializer = self.initializer, dtype = 'float32', name = 'Uc')
        self.bc = self.add_weight(shape = (self.output_dim),
                                          initializer = initializers.get('ones'), dtype = 'float32', name = 'bc')

        self.Wo = self.add_weight(shape = (self.input_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Wo')
        self.Uo = self.add_weight(shape = (self.output_dim, self.output_dim),
                                          initializer = self.initializer, dtype = 'float32', name = 'Uo')
        self.bo = self.add_weight(shape = (self.output_dim),
                                          initializer = initializers.get('ones'), dtype = 'float32', name = 'bo')


    def step(self, cell_blocks, cell_inputs):
        
        previous_hidden_state, previous_cell_state = array_ops.unstack(cell_blocks)
        current_states = cell_inputs


        input_gate = math_ops.sigmoid(math_ops.matmul(current_states, self.Wi) + math_ops.matmul(previous_hidden_state, self.Ui) + self.bi)
        
        forget_gate = math_ops.sigmoid(math_ops.matmul(current_states, self.Wf) + math_ops.matmul(previous_hidden_state, self.Uf) + self.bf)

        gate_weight = math_ops.tanh(math_ops.matmul(current_states, self.Wc) + math_ops.matmul(previous_hidden_state, self.Uc) + self.bc)

        current_cell_state = forget_gate * previous_cell_state + input_gate * gate_weight
        
        output_gate = math_ops.sigmoid(math_ops.matmul(current_states, self.Wo) + math_ops.matmul(previous_hidden_state, self.Uo) + self.bo)
        
        current_hidden_state = output_gate * math_ops.tanh(current_cell_state)
    
        return array_ops.stack([current_hidden_state, current_cell_state])
    

    def call(self, inputs, init_states):
        
        if not self.time_major:
            
            inputs = array_ops.transpose(inputs, [1, 0, 2])


        if self.backward:

            inputs = array_ops.reverse_sequence_v2(inputs, [inputs.shape[0]] * inputs.shape[1], seq_axis = 0, batch_axis = 1)
            
        outputs = functional_ops.scan(self.step, inputs, initializer = init_states) 
        outputs = array_ops.transpose(outputs, [1, 2, 0, 3])[0]


        if self.backward:
            
            #outputs = array_ops.reverse_sequence_v2(outputs[0], [outputs[0].shape[1]] * outputs[0].shape[0], seq_axis = 1, batch_axis = 0)
            outputs = array_ops.reverse_sequence_v2(outputs, [outputs.shape[1]] * outputs.shape[0], seq_axis = 1, batch_axis = 0)

        return outputs
    