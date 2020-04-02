import numpy as np

from keras import backend as K
from keras import activations
from keras.layers import Recurrent
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D
from keras.engine import InputSpec

class PreCNet(Recurrent):
    '''PreCNet - deep network based on predictive coding schema by Rao and Ballard
    see [PreCNet: Next Frame Video Prediction Based on Predictive Coding] by Straka et al. for details
    Code from PredNet - Lotter et al. 2016 used as a starting point for this code.

    # Arguments
        stack_sizes: number of channels in targets (A) and predictions (Ahat) in each layer of the architecture.
            Length is the number of layers in the architecture.
            First element is the number of channels in the input.
            Ex. (3, 16, 32) would correspond to a 3 layer architecture that takes in RGB images and has 16 and 32
                channels in the second and third layers, respectively.
        R_stack_sizes: number of channels in the representation (R) modules.
            Length must equal length of stack_sizes, but the number of channels per layer can be different.
            R_stack_sizes[i] must be equal stack_sizes[i+1] for i=0,1,..len(stack_sizes)-2
        Ahat_filt_sizes: filter sizes for the prediction (Ahat) modules.
            Has length equal to length of stack_sizes.
            Ex. (3, 3, 3) would mean that the predictions for each layer are computed by a 3x3 convolution of the
                representation (R) modules at each layer.
        R_filt_sizes: filter sizes for the representation (R) modules.
            Has length equal to length of stack_sizes.
            Corresponds to the filter sizes for all convolutions in the LSTM.
        pixel_max: the maximum pixel value.
            Used to clip the pixel-layer prediction.
        error_activation: activation function for the error (E) units.
        Ahat_activation: activation function for the prediction (A_hat) units.
        LSTM_activation: activation function for the cell and hidden states of the LSTM.
        LSTM_inner_activation: activation function for the gates in the LSTM.
        output_mode: either 'error', 'prediction', 'all' or layer specification (ex. R2, see below).
            Controls what is outputted by the PredNet.
            If 'error', the mean response of the error (E) units of each layer will be outputted.
                That is, the output shape will be (batch_size, nb_layers).
            If 'prediction', the frame prediction will be outputted.
            If 'all', the output will be the frame prediction concatenated with the mean layer errors.
                The frame prediction is flattened before concatenation.
                Nomenclature of 'all' is kept for backwards compatibility, but should not be confused with returning all of the layers of the model
            For returning the features of a particular layer, output_mode should be of the form unit_type + layer_number.
                For instance, to return the features of the LSTM "representational" units in the lowest layer, output_mode should be specificied as 'R0'.
                The possible unit types are 'Rtd', 'Rbu' 'Ahat', 'Atd', 'Abu' and 'Etd', 'Ebu' corresponding to the 'representation', 'prediction', 'target', and 'error' units respectively (td-top down, bu-bottom up).
        extrap_start_time: time step for which model will start extrapolating.
            Starting at this time step, the prediction from the previous time step will be treated as the "actual"
        data_format: 'channels_first' or 'channels_last'.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.

    # References
        -[PreCNet: Next Frame Video Prediction Based on Predictive Coding]
        -[Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects](http://www.nature.com/neuro/journal/v2/n1/pdf/nn0199_79.pdf)
        -[Deep predictive coding networks for video prediction and unsupervised learning](https://arxiv.org/abs/1605.08104)
        -[Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
        -[Convolutional LSTM network: a machine learning approach for precipitation nowcasting](http://arxiv.org/abs/1506.04214)
    '''

    def __init__(self, stack_sizes, R_stack_sizes, Ahat_filt_sizes, R_filt_sizes, pixel_max=1., error_activation='relu', Ahat_activation='relu',
                 LSTM_activation='tanh', LSTM_inner_activation='hard_sigmoid',
                 output_mode='error', extrap_start_time=None,
                 data_format=K.image_data_format(), **kwargs):
        self.stack_sizes = stack_sizes
        self.nb_layers = len(stack_sizes)
        assert len(R_stack_sizes) == self.nb_layers, 'len(R_stack_sizes) must equal len(stack_sizes)'
        self.R_stack_sizes = R_stack_sizes
        assert len(Ahat_filt_sizes) == self.nb_layers, 'len(Ahat_filt_sizes) must equal len(stack_sizes)'
        self.Ahat_filt_sizes = Ahat_filt_sizes
        assert len(R_filt_sizes) == (self.nb_layers), 'len(R_filt_sizes) must equal len(stack_sizes)'
        self.R_filt_sizes = R_filt_sizes

        for i in range(len(R_stack_sizes)-1):
            assert R_stack_sizes[i] == stack_sizes[i+1], "R_stack_sizes[i] must be equal stack_sizes[i+1] for i=0,1,..len(stack_sizes)-2"

        self.pixel_max = pixel_max
        self.error_activation = activations.get(error_activation)
        self.Ahat_activation = activations.get(Ahat_activation)
        self.LSTM_activation = activations.get(LSTM_activation)
        self.LSTM_inner_activation = activations.get(LSTM_inner_activation)

        default_output_modes = ['prediction', 'error', 'all']
        layer_output_modes = [layer + str(n) for n in range(self.nb_layers) for layer in ['Rtd', 'Etd', 'Ebu', 'Atd', 'Abu', 'Ahat']]
        layer_output_modes += [layer + str(n) for n in range(self.nb_layers - 1) for layer in ['Rbu']]
        assert output_mode in default_output_modes + layer_output_modes, 'Invalid output_mode: ' + str(output_mode)
        self.output_mode = output_mode
        if self.output_mode in layer_output_modes:
            self.output_layer_type = self.output_mode[:-1]
            self.output_layer_num = int(self.output_mode[-1])
        else:
            self.output_layer_type = None
            self.output_layer_num = None
        self.extrap_start_time = extrap_start_time

        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {channels_last, channels_first}'
        self.data_format = data_format
        self.channel_axis = -3 if data_format == 'channels_first' else -1
        self.row_axis = -2 if data_format == 'channels_first' else -3
        self.column_axis = -1 if data_format == 'channels_first' else -2
        super(PreCNet, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=5)]

    def compute_output_shape(self, input_shape):
        if self.output_mode == 'prediction':
            out_shape = input_shape[2:]
        elif self.output_mode == 'error':
            out_shape = (self.nb_layers,)
        elif self.output_mode == 'all':
            out_shape = (np.prod(input_shape[2:]) + self.nb_layers,)
        else:
            stack_str = 'R_stack_sizes' if (self.output_layer_type == 'Rtd' or self.output_layer_type == 'Rbu') else 'stack_sizes'
            stack_mult = 2 if (self.output_layer_type == 'Etd' or self.output_layer_type == 'Ebu') else 1
            out_stack_size = stack_mult * getattr(self, stack_str)[self.output_layer_num]
            out_nb_row = input_shape[self.row_axis] / 2**self.output_layer_num
            out_nb_col = input_shape[self.column_axis] / 2**self.output_layer_num

            if self.data_format == 'channels_first':
                out_shape = (out_stack_size, out_nb_row, out_nb_col)
            else:
                out_shape = (out_nb_row, out_nb_col, out_stack_size)

        if self.return_sequences:
            return (input_shape[0], input_shape[1]) + out_shape
        else:
            return (input_shape[0],) + out_shape

    def get_initial_state(self, x):
        input_shape = self.input_spec[0].shape
        init_nb_row = input_shape[self.row_axis]
        init_nb_col = input_shape[self.column_axis]

        base_initial_state = K.zeros_like(x)  # (samples, timesteps) + image_shape
        non_channel_axis = -1 if self.data_format == 'channels_first' else -2
        for _ in range(2):
            base_initial_state = K.sum(base_initial_state, axis=non_channel_axis)
        base_initial_state = K.sum(base_initial_state, axis=1)  # (samples, nb_channels)

        initial_states = []
        states_to_pass = ['r', 'c', 'e']
        nlayers_to_pass = {u: self.nb_layers for u in states_to_pass}

        for u in states_to_pass:
            for l in range(nlayers_to_pass[u]):
                ds_factor = 2 ** l
                nb_row = init_nb_row // ds_factor
                nb_col = init_nb_col // ds_factor
                if u in ['r', 'c']:
                    stack_size = self.R_stack_sizes[l]
                elif u == 'e':
                    stack_size = 2 * self.stack_sizes[l]

                elif u == 'ahat':
                    stack_size = self.stack_sizes[l]
                output_size = stack_size * nb_row * nb_col  # flattened size

                reducer = K.zeros((input_shape[self.channel_axis], output_size)) # (nb_channels, output_size)
                initial_state = K.dot(base_initial_state, reducer) # (samples, output_size)
                if self.data_format == 'channels_first':
                    output_shp = (-1, stack_size, nb_row, nb_col)
                else:
                    output_shp = (-1, nb_row, nb_col, stack_size)
                initial_state = K.reshape(initial_state, output_shp)
                initial_states += [initial_state]

        if K._BACKEND == 'theano':
            from theano import tensor as T
            # There is a known issue in the Theano scan op when dealing with inputs whose shape is 1 along a dimension.
            # In our case, this is a problem when training on grayscale images, and the below line fixes it.
            initial_states = [T.unbroadcast(init_state, 0, 1) for init_state in initial_states]

        if self.extrap_start_time is not None:
            initial_states += [K.variable(0, int if K.backend() != 'tensorflow' else 'int32')]  # the last state will correspond to the current timestep
        return initial_states

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.conv_layers = {c: [] for c in ['id', 'fd', 'cd', 'od', 'iu', 'fu', 'cu', 'ou', 'ahat']}

        for l in range(self.nb_layers):
            for c in ['id', 'fd', 'cd', 'od']:
                act = self.LSTM_activation if c == 'cd' else self.LSTM_inner_activation
                self.conv_layers[c].append(Conv2D(self.R_stack_sizes[l], self.R_filt_sizes[l], padding='same', activation=act, data_format=self.data_format))

            if l < self.nb_layers - 1:
                for c in ['iu', 'fu', 'cu', 'ou']:
                    act = self.LSTM_activation if c == 'cu' else self.LSTM_inner_activation
                    self.conv_layers[c].append(Conv2D(self.R_stack_sizes[l], self.R_filt_sizes[l], padding='same', activation=act, data_format=self.data_format))


            act = 'relu' if l == 0 else self.Ahat_activation
            self.conv_layers['ahat'].append(Conv2D(self.stack_sizes[l], self.Ahat_filt_sizes[l], padding='same', activation=act, data_format=self.data_format))


        self.upsample = UpSampling2D(data_format=self.data_format)
        self.pool = MaxPooling2D(data_format=self.data_format)

        self.trainable_weights = []
        nb_row, nb_col = (input_shape[-2], input_shape[-1]) if self.data_format == 'channels_first' else (input_shape[-3], input_shape[-2])
        for c in sorted(self.conv_layers.keys()):
            for l in range(len(self.conv_layers[c])):
                ds_factor = 2 ** l
                if c == 'ahat':
                    nb_channels = self.R_stack_sizes[l]
                elif c in ['id', 'fd', 'cd', 'od']:
                    if l == len(self.conv_layers[c])-1:
                        nb_channels = 2 * self.stack_sizes[l] + self.R_stack_sizes[l]
                    else:
                        nb_channels = 2 * self.stack_sizes[l+1] + self.R_stack_sizes[l]

                elif c in ['iu', 'fu', 'cu', 'ou']:
                    nb_channels = 2 * self.stack_sizes[l] + self.R_stack_sizes[l]


                in_shape = (input_shape[0], nb_channels, nb_row // ds_factor, nb_col // ds_factor)

                if self.data_format == 'channels_last': in_shape = (in_shape[0], in_shape[2], in_shape[3], in_shape[1])
                with K.name_scope('layer_' + c + '_' + str(l)):
                    self.conv_layers[c][l].build(in_shape)
                self.trainable_weights += self.conv_layers[c][l].trainable_weights

        self.states = [None] * self.nb_layers*3

        if self.extrap_start_time is not None:
            self.t_extrap = K.variable(self.extrap_start_time, int if K.backend() != 'tensorflow' else 'int32')
            self.states += [None] # [timestep]

    def step(self, a, states):
        r_tm1 = states[:self.nb_layers]
        c_tm1 = states[self.nb_layers:2*self.nb_layers]
        e_tm1 = states[2*self.nb_layers:3*self.nb_layers]

        if self.extrap_start_time is not None:
            t = states[-1]

        a0=a[:]


        c = []
        r = []
        e = []

        ahat_list=[]

        # Update R units starting from the top
        for l in reversed(range(self.nb_layers)):
            if l == self.nb_layers - 1:
                inputs = [r_tm1[l], e_tm1[l]]
            else:
                inputs = [r_tm1[l], ed]


            inputs = K.concatenate(inputs, axis=self.channel_axis)
            id = self.conv_layers['id'][l].call(inputs)
            fd = self.conv_layers['fd'][l].call(inputs)
            od = self.conv_layers['od'][l].call(inputs)
            _c = fd * c_tm1[l] + id * self.conv_layers['cd'][l].call(inputs)
            _r = od * self.LSTM_activation(_c)
            c.insert(0, _c)
            r.insert(0, _r)

            ahat = self.conv_layers['ahat'][l].call(r[0])
            if l == 0:
                ahat = K.minimum(ahat, self.pixel_max)
                frame_prediction = ahat
            ahat_list.insert(0,ahat)

            if l > 0:
                a = self.pool.call(r_tm1[l-1])  # target for next layer
            else:
                if self.extrap_start_time is not None:
                    a = K.switch(t >= self.t_extrap, ahat, a0)
                else:
                    a = a0

            # compute errors
            e_up = self.error_activation(ahat - a)
            e_down = self.error_activation(a - ahat)




            e.insert(0, K.concatenate((e_up, e_down), axis=self.channel_axis))

            if l > 0:
                ed = self.upsample.call(e[0])

            if self.output_layer_num == l:
                if self.output_layer_type == 'Atd':
                    output = a
                if self.output_layer_type == 'Ahat':
                    output = ahat
                elif self.output_layer_type == 'Rtd':
                    output = r[0]
                elif self.output_layer_type == 'Etd':
                    output = e[0]

        # Update R units starting from the bottom
        for l in range(self.nb_layers):

            if l == 0:
                pass
            else:
                a = self.pool.call(r[l-1])
                ahat = ahat_list[l]
                e_up = self.error_activation(ahat - a)
                e_down = self.error_activation(a - ahat)
                e[l] = K.concatenate((e_up, e_down), axis=self.channel_axis)

            if l<self.nb_layers-1:
                inputs = [r[l], e[l]]

                inputs = K.concatenate(inputs, axis=self.channel_axis)
                iu = self.conv_layers['iu'][l].call(inputs)
                fu = self.conv_layers['fu'][l].call(inputs)
                ou = self.conv_layers['ou'][l].call(inputs)
                _c = fu * c[l] + iu * self.conv_layers['cu'][l].call(inputs)
                _r = ou * self.LSTM_activation(_c)
                c[l] = _c
                r[l] = _r


            if self.output_layer_num == l:
                if self.output_layer_type == 'Abu':
                    output = a
                elif self.output_layer_type == 'Rbu':
                    output = r[l]
                elif self.output_layer_type == 'Ebu':
                    output = e[l]



        if self.output_layer_type is None:
            if self.output_mode == 'prediction':
                output = frame_prediction
            else:
                for l in range(self.nb_layers):
                    layer_error = K.mean(K.batch_flatten(e[l]), axis=-1, keepdims=True)
                    all_error = layer_error if l == 0 else K.concatenate((all_error, layer_error), axis=-1)
                if self.output_mode == 'error':
                    output = all_error
                else:
                    output = K.concatenate((K.batch_flatten(frame_prediction), all_error), axis=-1)


        states = r + c + e
        if self.extrap_start_time is not None:
            states += [t + 1]
        return output, states

    def get_config(self):
        config = {'stack_sizes': self.stack_sizes,
                  'R_stack_sizes': self.R_stack_sizes,
                  'Ahat_filt_sizes': self.Ahat_filt_sizes,
                  'R_filt_sizes': self.R_filt_sizes,
                  'pixel_max': self.pixel_max,
                  'error_activation': self.error_activation.__name__,
                  'Ahat_activation': self.Ahat_activation.__name__,
                  'LSTM_activation': self.LSTM_activation.__name__,
                  'LSTM_inner_activation': self.LSTM_inner_activation.__name__,
                  'data_format': self.data_format,
                  'extrap_start_time': self.extrap_start_time,
                  'output_mode': self.output_mode}
        base_config = super(PreCNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
