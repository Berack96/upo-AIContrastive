from keras import layers, models
import numpy as np

class CustomModels:
    """
    Class to build and save custom models.
    """

    def __init__(self, latent_space:int, shape:tuple[int, int], directory:str):
        """
        Constructor to initialize the class.
        :param latent_space: The size of the latent space.
        :param shape: The shape of the input images.
        :param filename: complete path of the model file.
        """
        self.latent_space = latent_space
        self.shape = shape
        self.directory = directory

    def get_full_path(self, model_name:str):
        """
        Function to get the full path of the model.
        :param model_name: The name of the model.
        :return: Full path of the model.
        """
        model_path = f"{self.directory}/{model_name}_{self.latent_space}_{self.shape[0]}x{self.shape[1]}"
        history_path = model_path + "_history.npy"
        model_path = model_path + ".keras"
        return model_path, history_path

    def save_model_and_history(self, model:models.Model, history):
        """
        Function to save the model and training history.
        :param model: The model to be saved.
        :param history: The training history to be saved.
        """
        model_path, history_path = self.get_full_path(model.name)
        model.save(model_path)
        np.save(history_path, history.history)

    def load_history_of(self, model:models.Model):
        """
        Function to load the training history of the model.
        :param model: The model whose history is to be loaded.
        :return: The training history.
        """
        _, history_path = self.get_full_path(model.name)
        return np.load(history_path, allow_pickle=True).item()

    def autoencoder_build(self, vae:bool=False):
        """
        Function to get the autoencoder model.
        If the model is already trained, it loads the model and encoder/decoder layers.
        If not, it creates a new model and trains it.
        :param vae: If True, it creates a VAE model.
        :return: Encoder and decoder models.
        """
        try:
            model_path, _ = self.get_full_path("autoencoder" + ("-vae" if vae else ""))
            autoencoder = models.load_model(model_path)
            should_train = False
        except Exception as e:
            in_encoder = layers.Input(shape=(*self.shape, 1), dtype='float32')
            x = layers.BatchNormalization()(in_encoder)
            x = layers.Conv2D(32, 3, padding='same', strides=2)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Conv2D(64, 3, padding='same', strides=2)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Conv2D(128, 3, padding='same', strides=2)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Conv2D(256, 3, padding='same', strides=2)(x)
            before_flatten = x.shape[1:]
            x = layers.Flatten()(x)
            flatten = x.shape[1]
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            latent = layers.Dense(self.latent_space, activation='sigmoid')(x)
            encoder = models.Model(in_encoder, latent, name='encoder')

            in_decoder = layers.Input(shape=(self.latent_space,))
            x = layers.Dense(flatten)(in_decoder)
            x = layers.LeakyReLU()(x)
            x = layers.Reshape(before_flatten)(x)
            x = layers.Conv2DTranspose(256, 3, padding='same', strides=2)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Conv2DTranspose(128, 3, padding='same', strides=2)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Conv2DTranspose(64, 3, padding='same', strides=2)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Conv2DTranspose(32, 3, padding='same', strides=2)(x)
            x = layers.LeakyReLU()(x)
            out_decoder = layers.Conv2DTranspose(1, 3, padding='same', activation='sigmoid')(x)
            decoder = models.Model(in_decoder, out_decoder, name='decoder')

            in_autoencoder = layers.Input(shape=(*self.shape, 1))
            encoder_out = encoder(in_autoencoder)
            decoder_out = decoder(encoder_out)
            autoencoder = models.Model(in_autoencoder, decoder_out, name='autoencoder')
            autoencoder.compile(optimizer='adam', loss='mse')
            should_train = True

        return should_train, autoencoder

    def classifier_build(self, total_classes: int):
        """
        Function to get the classifier model.
        If the model is already trained, it loads the model.
        If not, it creates a new model and trains it.
        :param latent_space: The size of the latent space.
        :param total_classes: The number of classes.
        :param filename: complete path of the model file.
        :return: Classifier model.
        """
        try:
            model_path, _ = self.get_full_path("classifier")
            classifier = models.load_model(model_path)
            should_train = False
        except Exception:
            in_classifier = layers.Input(shape=(self.latent_space,), dtype='float32')
            x = layers.BatchNormalization()(in_classifier)
            x = layers.Dense(128, use_bias=False)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dense(32, use_bias=False)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dense(8, use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            out_classifier = layers.Dense(total_classes, activation='softmax')(x)
            classifier = models.Model(in_classifier, out_classifier, name='classifier')
            classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            should_train = True

        return should_train, classifier
