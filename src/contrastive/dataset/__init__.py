import logging
import numpy as np
import numpy.typing as npt



log = logging.getLogger(__name__)
DATASET_ROOT = "datasets/"

class DatasetCachedData:
    '''
    Class to hold dataset embeddings.
    Attributes:
        path (str): Path to the dataset file.
        x_train (np.ndarray): Training data features.
        y_train (np.ndarray): Training data labels.
        x_val (np.ndarray): Validation data features.
        y_val (np.ndarray): Validation data labels.
        x_test (np.ndarray): Test data features.
        y_test (np.ndarray): Test data labels.
        latent_space (int): Dimensionality of the latent space.
    '''
    def __init__(self, path: str):
        self.path = path
        self.x_train: npt.NDArray[np.float64] = None  # type: ignore
        self.y_train: npt.NDArray[np.int_] = None  # type: ignore
        self.x_val: npt.NDArray[np.float64] = None  # type: ignore
        self.y_val: npt.NDArray[np.int_] = None  # type: ignore
        self.x_test: npt.NDArray[np.float64] = None  # type: ignore
        self.y_test: npt.NDArray[np.int_] = None  # type: ignore
        self.latent_space: int = 0

    def load(self):
        dataset = np.load(self.path, allow_pickle=True)
        self.x_train = dataset['x_train']
        self.y_train = dataset['y_train']
        self.x_val = dataset['x_val']
        self.y_val = dataset['y_val']
        self.x_test = dataset['x_test']
        self.y_test = dataset['y_test']
        self.latent_space = int(self.x_train.shape[1])

    def save(self):
        np.savez_compressed(
            self.path,
            x_train=self.x_train,
            y_train=self.y_train,
            x_val=self.x_val,
            y_val=self.y_val,
            x_test=self.x_test,
            y_test=self.y_test
        )


class Dataset:
    '''
    Base class for datasets.
    Attributes:
        name (str): Name of the dataset.
    '''
    def __init__(self, name: str):
        self.name = name
        self.cache = DATASET_ROOT + self.name + "_embeddings.npz"
        self.data: DatasetCachedData | None = None

    def load_data_cached(self) -> bool:
        '''
        Load the dataset in memory.
        This method will get the data from cache.
        if the data is cached, it will load it into memory and return True.
        Returns:
            bool: True if data was loaded from cache, False otherwise.
        '''
        try:
            self.data = DatasetCachedData(self.cache)
            self.data.load()
            log.info(f"Loaded dataset '{self.name}' from cache.")
            return True
        except FileNotFoundError:
            log.error(f"Cache file for dataset '{self.name}' not found.")
            self.data = self.build_data()
            self.save_data_cached()
            return False

    def save_data_cached(self):
        '''
        Save the dataset to cache.
        This method will save the data to cache for future use.
        '''
        if self.data is None:
            log.error("No data to save to cache.")
            return
        self.data.save()
        log.info(f"Saved dataset '{self.name}' to cache.")

    def build_data(self) -> DatasetCachedData:
        '''
        Build the dataset.
        This method should be implemented by subclasses to create the dataset.
        '''
        raise NotImplementedError("Subclasses should implement this method.")
