from tensorflow import keras


class GetLSTMStatesLayer(keras.layers.Layer):
    """Function that retrieves the long & short term state of an
    LSTM unit (when return_state=True)
    Note: The first entry of the result list is the output, which is
    equivalent to the short-term state (hence it is dropped)
    """

    def __init__(self):
        super(GetLSTMStatesLayer, self).__init__()

    def call(self, input, **kwargs):
        return keras.layers.Concatenate()(input[1:])

