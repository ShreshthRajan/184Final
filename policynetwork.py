import numpy as np
import tensorflow as tf

class PolicyNetwork:
    def __init__(self, board_size=7):
        input1 = tf.keras.layers.Input(shape=(board_size, board_size,1))
        x1 = tf.keras.layers.Conv2D(32, (4, 4), activation='relu', use_bias= True,
                           kernel_initializer= tf.keras.initializers.RandomNormal(mean=0.0, stddev=2),
                           input_shape=(board_size, board_size, 1), padding='same',
                           bias_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=2))(input1)
        x2 = tf.keras.layers.Conv2D(64, (4, 4), activation='relu', kernel_initializer= tf.keras.initializers.RandomNormal(mean=0.0, stddev=2),
                           use_bias= True, bias_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=2),
                                    padding='same')(x1)
        x3 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
        x4 = tf.keras.layers.Conv2D(128, (4, 4), use_bias=True,
                               bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=2),
                               activation='relu',
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=2),
                               padding='same')(x3)
        x5 = tf.keras.layers.MaxPooling2D((2, 2))(x4)
        x6 = tf.keras.layers.Flatten()(x5)
        x7 = tf.keras.layers.Dense(128, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=2),
                              use_bias=True, bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=2),
                              activation='relu')(x6)
        x8 = tf.keras.layers.Dense(board_size ** 2, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=2),
                              use_bias=True, bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=2),
                              activation='softmax')(x7)
        x_init = tf.keras.layers.Flatten()(input1)
        x_mask = NonZeroMaskingLayer()([x_init, x8])
        output = CustomScaler()(x_mask)
        self.model = tf.keras.models.Model(inputs = input1, outputs = output)
        self.board_size = board_size


    def __deepcopy__(self, memo):
        new_instance = PolicyNetwork(board_size=self.board_size)
        new_instance.model = tf.keras.models.clone_model(self.model)
        return new_instance

    def compute_gradient(self, state, action):
        '''
        :param state: (board_size, board_size) numpy array
        :param action: integer
        :return: the gradient of log p_theta(action|state)
        '''
        with tf.GradientTape() as tape:
            input_tensor = tf.constant(state.reshape(1, self.board_size, self.board_size, 1), dtype=tf.float64)
            tape.watch(input_tensor)
            output_probabilities = self.model(input_tensor)
            value_at_index = tf.math.log(output_probabilities[:, action])
        return tape.gradient(value_at_index, self.model.trainable_variables)


    def update(self, gradients, learning_rate = 0.01):
        '''
        :param gradients: A tensorflow gradient
        :param learning_rate:
        :return:
        '''
        if gradients:
            transposed_gradients = list(zip(*gradients))
            column_sums = [sum(col) for col in transposed_gradients]
            mean_gradient = [sum_col / len(gradients) for sum_col in column_sums]
            for param, grad in zip(self.model.trainable_variables, mean_gradient):
                param.assign_sub(learning_rate * grad)


    def sample(self, state):
        '''
        :param state: np.array with shape (board_size, board_size)
        :return: Action
        '''
        s = state.reshape(self.board_size, self.board_size, 1)
        p = self.model.predict(np.expand_dims(s, axis=0))

        return np.argmax(p)

class NonZeroMaskingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(NonZeroMaskingLayer, self).__init__()

    def call(self, inputs):
        x_init, x7 = inputs
        x7 = tf.reshape(x7, shape=(-1, 49, 1))
        mask = tf.cast(tf.math.equal(x_init, 0.0), tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        masked_x7 = tf.keras.layers.Flatten()(x7 * mask)
        return masked_x7

class CustomScaler(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomScaler, self).__init__()

    def call(self, inputs):
        zero_mask = tf.cast(tf.equal(inputs, 0), tf.float32)
        non_zero_sum = tf.reduce_sum(inputs * (1 - zero_mask), axis=-1, keepdims=True)
        scaled_non_zero = inputs * (1 - zero_mask) / non_zero_sum
        return scaled_non_zero + zero_mask * inputs

