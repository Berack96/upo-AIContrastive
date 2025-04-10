import numpy as np
import pandas as pd

from tqdm import tqdm
from keras.api.preprocessing.image import load_img, img_to_array

def load_cache(file) -> dict:
    """
    Loads a cache file if it exists, otherwise returns an empty dictionary.
    :param file: Path to the cache file.
    :return: Dictionary containing the loaded data.
    """
    try:
        return dict(np.load(file, allow_pickle=True).item())
    except Exception:
        return {}

def images_to_numpy(images: pd.Series, size: tuple[int, int]):
    """
    Converts a list of image file paths to a NumPy array of images.
    :param images: Series of image file paths.
    :param size: Tuple specifying the target size of the images.
    :return: NumPy array of images.
    """
    np_images = np.zeros((len(images), *size, 1), dtype=np.uint8)
    for i, img_name in enumerate(tqdm(images)):
        img = load_img(img_name, target_size=size, color_mode='grayscale')
        np_images[i] = img_to_array(img, dtype=np.uint8).reshape((*size, 1))
    return np_images

def classes_to_numpy(classes: pd.Series):
    """
    Converts a list of class labels to a NumPy array of integer labels.
    :param classes: Series of class labels.
    :return: NumPy array of integer labels.
    """
    return np.array(pd.factorize(classes)[0])

def covid_cxr_data(directory:str, size:tuple[int, int]=(256, 256), cache_name:str='cache'):
    """
    Loads the COVID-19 CXR dataset and caches it if not already cached.
    :param directory: Directory containing the dataset.
    :param size: Tuple specifying the target size of the images.
    :param cache_name: Name of the cache file.
    :return: Tuple containing training, validation, and test datasets.
    """
    directory = f"{directory}/covid_cxr"
    cache = f"{directory}/{cache_name}_{size[0]}x{size[1]}.npy"
    dataset = load_cache(cache)

    if len(dataset) == 0:
        types = ['train', 'val', 'test']
        all_files = []
        for t in types:
            df = pd.read_csv(f"{directory}/{t}.txt", delimiter=' ', header=None)
            df[1] = df[1].apply(lambda x: f"{directory}/{t}/{x}")
            all_files.append(df)

        df = pd.concat(all_files)
        df.columns = ['id', 'filename', 'class', 'source']
        images = images_to_numpy(df['filename'], size)
        predictions = classes_to_numpy(df['class'])

        train_tot = len(all_files[0])
        val_tot = train_tot + len(all_files[1])
        test_tot = val_tot + len(all_files[2])

        dataset['train'] = (images[:train_tot], predictions[:train_tot])
        dataset['val'] = (images[train_tot:val_tot], predictions[train_tot:val_tot])
        dataset['test'] = (images[val_tot:test_tot], predictions[val_tot:test_tot])

        np.save(cache, dataset)

    x_train, y_train = dataset['train'][0], dataset['train'][1]
    x_val, y_val = dataset['val'][0], dataset['val'][1]
    x_test, y_test = dataset['test'][0], dataset['test'][1]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def load_or_predict_values(directory:str, model , x_train:np.ndarray, x_val:np.ndarray, x_test:np.ndarray, batch_predictions:int, cache_name:str='cache_predictions'):
    """
    Loads or predicts values using the given model.
    :param model: Keras model to use for predictions.
    :param x_train: Training data.
    :param x_val: Validation data.
    :param x_test: Test data.
    :param batch_predictions: Batch size for predictions.
    :return: Tuple containing encoded training, validation, and test data.
    """
    shape = x_train.shape[1:]
    latent_space = model.output_shape[1]
    cache = f"{directory}/covid_cxr/{cache_name}_{shape[0]}x{shape[1]}_{latent_space}.npy"
    predictions = load_cache(cache)

    if len(predictions) <= 0:
        batch_train_steps = len(x_train) // batch_predictions + 1
        batch_val_steps = len(x_val) // batch_predictions + 1
        batch_test_steps = len(x_test) // batch_predictions + 1

        gen_train = data_generator(x_train, batch_predictions)
        gen_val = data_generator(x_val, batch_predictions)
        gen_test = data_generator(x_test, batch_predictions)

        x_train_encoded = model.predict(gen_train, steps=batch_train_steps)
        x_val_encoded = model.predict(gen_val, steps=batch_val_steps)
        x_test_encoded = model.predict(gen_test, steps=batch_test_steps)

        predictions['train'] = x_train_encoded
        predictions['val'] = x_val_encoded
        predictions['test'] = x_test_encoded
        np.save(cache, predictions)

    return predictions['train'], predictions['val'], predictions['test']

def data_generator(x, batch_size):
    """
    Generator function to yield batches of data.
    :param x: Input data.
    :param batch_size: Size of each batch.
    :return: Yields batches of data.
    """
    num_samples = x.shape[0]
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices)  # Mescola gli indici all'inizio di ogni epoca
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = image_int_to_float(x[batch_indices])
            yield (batch, )

def data_generator_autoencoder(x, batch_size):
    """
    Generator function to yield batches of data for autoencoder.
    :param x: Input data.
    :param batch_size: Size of each batch.
    :return: Yields batches of data.
    """
    for batch in data_generator(x, batch_size):
        yield (batch[0], batch[0])

def image_int_to_float(image:np.ndarray):
    """
    Converts an image from integer to float format.
    :param image: Input image.
    :return: Converted image.
    """
    return image.astype(np.float32) / 255.0
