import numpy as np
import tensorflow as tf

class ValueNetwork:
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
        output = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=2),
                              use_bias=True, bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=2),
                              activation='tanh')(x7)
        self.model = tf.keras.models.Model(inputs = input1, outputs = output)
        self.board_size = board_size


    def __deepcopy__(self, memo):
        new_instance = ValueNetwork(board_size=self.board_size)
        new_instance.model = tf.keras.models.clone_model(self.model)
        return new_instance

    def compute_gradient(self, state, final_result):
        '''
        :param state: (board_size, board_size) numpy array
        :param final_result: 1 and -1
        :return: the square difference between estimated value and the final result
        '''
        with tf.GradientTape() as tape:
            input_tensor = tf.constant(state.reshape(1, self.board_size, self.board_size, 1), dtype=tf.float64)
            tape.watch(input_tensor)
            output_value = self.model(input_tensor)
            tmp = output_value - final_result
            gradient = [grad * tmp for grad in tape.gradient(output_value,self.model.trainable_variables)]
        return gradient


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


