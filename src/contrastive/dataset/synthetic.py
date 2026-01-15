import logging
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import normalize # type: ignore
from . import Dataset, DatasetCachedData

log = logging.getLogger(__name__)



class SyntheticParameters:
    '''
    Parameters for synthetic dataset generation.
    Attributes:
        num_samples (int): Total number of samples in the dataset.
        classes (int): Number of distinct classes.
        output_embedding_dim (int): Dimension of the output embeddings.
        latent_dim (int): Dimensionality of the shared latent space where classes are defined.
        noise_scale (float): Scale of the Gaussian noise added to the samples.
        similarity (float): Controls how similar the prototypical samples of different classes are.
        size_train (int): Number of training samples.
        size_val (int): Number of validation samples.
        size_test (int): Number of test samples.
        seed (int): Random seed for reproducibility.
    '''
    def __init__(self,
                 num_samples: int = 1000,
                 classes: int = 2,
                 output_embedding_dim: int = 1024,
                 latent_dim: int = 2,
                 noise_scale: float = 0.4,
                 similarity: float = 0.01,
                 split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
                 seed: int = 42
                 ):
        assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"
        assert all(ratio > 0 for ratio in split_ratios), "All split ratios must be positive"
        assert num_samples > 0, "Number of samples must be positive"
        assert classes > 0, "Number of classes must be positive"
        assert output_embedding_dim > 0, "Output embedding dimension must be positive"
        assert latent_dim > 0, "Latent dimension must be positive"
        assert noise_scale >= 0, "Noise scale must be non-negative"
        assert similarity >= 0, "Similarity must be non-negative"

        self.num_samples = num_samples
        self.classes = classes
        self.output_embedding_dim = output_embedding_dim
        self.latent_dim = latent_dim
        self.noise_scale = noise_scale
        self.similarity = similarity
        self.index_end_train = int(self.num_samples * split_ratios[0])
        self.index_end_val = int(self.num_samples * split_ratios[1]) + self.index_end_train
        self.seed = seed

class SyntheticDataset(Dataset):
    def __init__(self, parameters: SyntheticParameters=SyntheticParameters()):
        super().__init__("synthetic")
        self.params = parameters

    def build_data(self) -> DatasetCachedData:
        p = self.params
        np.random.seed(p.seed) # for reproducibility

        # 1. Create class prototypes in a shared latent space
        # These vectors represent the "pure concept" of each class.
        log.info(f"Creating {p.classes} class prototypes in a {p.latent_dim}-dimensional latent space.")
        prototypes = self._create_class_prototypes(p.classes, p.latent_dim, p.similarity)
        
        # 2. Create separate projection matrices for text and image spaces
        # These matrices will map the shared concepts into their respective, 
        #different vector spaces.
        projection = np.random.randn(p.latent_dim, p.output_embedding_dim)

        # 3. Project latent prototypes to create class centroids in each space
        class_centroids = prototypes @ projection

        # 4. Generate individual samples for each class
        log.info(f"Generating {p.num_samples} samples with noise scale {p.noise_scale}.")
        emb, labels = self._generate_samples(class_centroids, p.num_samples, p.noise_scale)

        # 5. Split the dataset into training, validation, and test sets
        train_data = (emb[:p.index_end_train], labels[:p.index_end_train])
        val_data = (emb[p.index_end_train:p.index_end_val], labels[p.index_end_train:p.index_end_val])
        test_data = (emb[p.index_end_val:], labels[p.index_end_val:])

        # 6. Store in DatasetCachedData
        dataset = DatasetCachedData(self.cache)
        dataset.x_train, dataset.y_train = train_data
        dataset.x_val, dataset.y_val = val_data
        dataset.x_test, dataset.y_test = test_data
        dataset.latent_space = p.latent_dim
        return dataset

    @staticmethod
    def _create_class_prototypes(num_classes: int, latent_dim: int, similarity: float) -> npt.NDArray[np.float64]:
        '''
        Create class prototypes in a shared latent space.
        Args:
            num_classes (int): Number of distinct classes.
            latent_dim (int): Dimensionality of the shared latent space.
            similarity (float): Controls how similar the prototypical samples of different classes are.
        Returns:
            NDArray[np.float64]: Array of shape (num_classes, latent_dim) representing class prototypes.
        '''
        latent_prototypes = np.random.randn(num_classes, latent_dim)
        base_prototype = np.random.randn(latent_dim)
        for i in range(num_classes):
            variation = latent_prototypes[i] * similarity
            latent_prototypes[i] = base_prototype + variation

        latent_prototypes /= np.linalg.norm(latent_prototypes, axis=1, keepdims=True)
        return latent_prototypes

    def _generate_samples(self, class_centroids: npt.NDArray[np.float64], num_samples: int, noise_scale: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]:
        '''
        Generate individual samples for each class by adding Gaussian noise to class centroids.
        Args:
            class_centroids (np.ndarray): Array of shape (num_classes, output_embedding_dim) representing class centroids.
            num_samples (int): Total number of samples to generate.
            noise_scale (float): Scale of the Gaussian noise added to the samples.
        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]: Tuple containing:
                - samples (npt.NDArray[np.float64]): Array of shape (num_samples, output_embedding_dim) representing
                    the generated samples.
                - labels (npt.NDArray[np.int_]): Array of shape (num_samples,) representing the class labels for each sample.
        '''
        emb_list: list[npt.NDArray[np.float64]] = []
        labels_list: list[int] = []
        num_classes = class_centroids.shape[0]
        output_embedding_dim = class_centroids.shape[1]

        for class_idx in np.random.randint(0, num_classes, size=num_samples):
            centroid = class_centroids[class_idx]
            noise = np.random.normal(0, noise_scale, size=output_embedding_dim)
            final_embedding = centroid + noise

            emb_list.append(final_embedding)
            labels_list.append(class_idx)

        # Normalize embeddings to unit length
        emb = normalize(np.array(emb_list))  # type: ignore
        labels = np.array(labels_list)
        return emb, labels  #type: ignore
