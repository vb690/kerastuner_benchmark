from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, PReLU, ReLU, ELU
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

from kerastuner import HyperModel


def create_dense_block(input_tensor, tag, batch_norm, hp):
    '''
    Function creating a tunable block of densely connected
    layers
    '''
    ACTIVATIONS = {
        'relu': ReLU,
        'elu': ELU,
        'lelu': LeakyReLU,
        'prelu': PReLU
    }
    for layer in range(hp.Int(
                        'layers_{}'.format(tag),
                        min_value=1,
                        max_value=10
                        )
                       ):

        if layer == 0:
            dense = Dense(
                units=hp.Int(
                    name='units_layer_{}_{}'.format(layer, tag),
                    min_value=32,
                    max_value=512,
                    step=32
                )
            )(input_tensor)

        else:
            dense = Dense(
                units=hp.Int(
                    'units_layer_{}_{}'.format(layer, tag),
                    min_value=32,
                    max_value=512,
                    step=32
                )
            )(dense)
        if batch_norm:
            dense = BatchNormalization()(dense)
        chosen_activation = hp.Choice(
            'activation_layer_{}_{}'.format(layer, tag),
            ['lelu', 'relu', 'elu']
        )
        dense = Activation(
            ACTIVATIONS[chosen_activation]()
        )(dense)
        dense = Dropout(
            rate=hp.Float(
                'dropout_layer_{}_{}'.format(layer, tag),
                min_value=0.0,
                max_value=0.4,
                step=0.1
            )
        )(dense)
    return dense


class MultiLayerPerceptron(HyperModel):
    def __init__(self, X_shape, y_shape):
        self.X_shape = X_shape
        self.y_shape = y_shape

    def build(self, hp):
        '''
        Create a tunable MLP
        '''
        batch_norm = hp.Boolean(
            name='batch_norm'
        )

        input_tensor = Input(
            shape=(self.X_shape[1],)
        )
        hyperblock = create_dense_block(
            input_tensor=input_tensor,
            tag='dense_block',
            batch_norm=batch_norm,
            hp=hp
        )
        out = Dense(
                units=self.y_shape[1]
            )(hyperblock)

        out = Activation('softmax')(out)

        loss = 'categorical_crossentropy'
        metric = 'accuracy'

        model = Model(input_tensor, out)
        model.compile(
            optimizer=Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-5,
                    max_value=1,
                    sampling='log'
                )
            ),
            loss=loss,
            metrics=[metric]

        )
        return model
