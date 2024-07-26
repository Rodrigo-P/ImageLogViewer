import torchvision.transforms.v2 as T
import SimpleITK as sitk
import numpy as np
import radiomics
import torch
import cv2
import six

from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Optional, Union
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from skimage import measure
from torch import Tensor


def broken_stick(X: np.ndarray, normalize: bool = False) -> int:
    """
    Calculate the number of principal components to retain using the broken stick method.

    Args:
        X (np.ndarray): The input data matrix of shape (n_samples, n_features).
        normalize (bool, optional): Whether to normalize the data. Defaults to False.

    Returns:
        int: The number of principal components to retain.

    """
    if normalize:
        # Z-normalize the data
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Initialize and fit PCA
    pca = PCA()
    pca.fit(X)

    # Calculate the explained variance
    explained_variance = pca.explained_variance_ratio_
    expected_values: list[float] = []
    d: int = len(explained_variance)
    c: int = 1  # TODO: Discuss with the team about the value of c

    # Calculate the expected values
    for i in range(1, d + 1):
        j: int = d - i + 1

        total_sum: float = 0.0
        for k in range(1, (d - j + 1) + 1):
            total_sum += 1 / (d + 1 - k)
        total_sum *= c / d

        expected_values.append(total_sum)

    # Compare the expected values with the explained variance

    def compare_greater(x, y):
        return x > y

    def compare_less(x, y):
        return x < y

    func_compare: Union[Callable | None] = None
    if expected_values[0] > explained_variance[0]:
        func_compare = compare_greater
    else:
        func_compare = compare_less

    ans: int = 1
    while (
        func_compare(expected_values[ans - 1], explained_variance[ans - 1]) and ans != d
    ):
        ans += 1

    return max(2, ans)


class SLICSegmentation(torch.nn.Module):
    """A PyTorch module that applies SLIC to an image."""

    def __init__(
        self,
        slic_algorithm: int = cv2.ximgproc.SLIC,
        region_size: int = 10,
        ruler: float = 10.0,
        slic_iterations: int = 10,
        gaussian_blur: Optional[Union[tuple[int, int], int]] = 3,
        sigma: Union[tuple[float, float], float] = 0.0,
        enforce_connectivity: bool = True,
        min_element_size: int = 25,
        return_labels: bool = True,
        labels_new_dimension: bool = False,
        return_slic_object: bool = False,
    ) -> None:
        """Initialize the SLICSegmentation class.

        Args:
            slic_algorithm (int): The algorithm to use for SLIC.
                Options: cv2.ximgproc.SLIC, cv2.ximgproc.SLICO, cv2.ximgproc.MSLIC.
                Default is cv2.ximgproc.SLIC.
            region_size (int): The average superpixel size measured in pixels.
                Ignored if slic_algorithm is cv2.ximgproc.SLICO or cv2.ximgproc.MSLIC.
                Default is 10.
            ruler (float): The enforcement of superpixel smoothness factor of superpixel.
                Default is 10.0.
            slic_iterations (int): The number of iterations.
                Default is 10.
            gaussian_blur (Optional[Union[tuple[int, int], int]]): The kernel size for Gaussian blur.
                If None, no Gaussian blur is applied.
                Default is 3.
            sigma (tuple[float, float] | float): The standard deviation for Gaussian blur.
                If gaussian_blur is None, this parameter is ignored.
                Default is 0.0.
            enforce_connectivity (bool): If True, enforce connectivity between superpixels.
                Default is True.
            min_element_size (int): The minimum element size in percents that should be
                absorbed into a bigger superpixel. Ignored if enforce_connectivity is False.
                Default is 25.
            return_labels (bool): If True, return the superpixel labels.
                Mutually exclusive with return_slic_object.
                At least one of return_labels and return_slic_object must be True.
                Default is True.
            labels_new_dimension (bool): If True, add a new dimension to the labels.
                Ignored if return_labels is False.
                Default is False.
            return_slic_object (bool): If True, return the SLIC object.
                Mutually exclusive with return_labels.
                At least one of return_labels and return_slic_object must be True.
                Default is False.
        """
        super().__init__()

        self.slic_algorithm: int = slic_algorithm
        self.region_size: int = region_size
        self.ruler: float = ruler
        self.slic_iterations: int = slic_iterations
        self.gaussian_blur: Optional[tuple[int, int]] = (
            (gaussian_blur, gaussian_blur)
            if isinstance(gaussian_blur, int)
            else gaussian_blur
        )
        self.sigma: tuple[float, float] = (
            (sigma, sigma) if isinstance(sigma, float) else sigma
        )
        self.enforce_connectivity: bool = enforce_connectivity
        self.min_element_size: int = min_element_size
        self.return_labels: bool = return_labels
        self.labels_new_dimension: bool = labels_new_dimension
        self.return_slic_object: bool = return_slic_object
        self.pil_transform = T.ToPILImage()
        self.tensor_transform = T.ToImage()
        self._validate_args()

    def forward(
        self, image: Tensor
    ) -> Union[Tensor, list[cv2.ximgproc_SuperpixelSLIC]]:
        """Applies SLIC to the input tensor image.

        The method is individually applied to each channel of the input image.

        Args:
            image (torch.Tensor): The input tensor image.
                Shape should be (C, height, width).

        Returns:
            Tensor | list[cv2.ximgproc_SuperpixelSLIC]: The resulting tensor image mask
                or the SLIC object for each channel.
        """
        # Convert tensor to numpy array for each channel
        np_image: np.ndarray = np.array(
            [
                self.pil_transform(image[i].unsqueeze(0)).convert("L")
                for i in range(image.shape[0])
            ]
        )
        # Apply SLIC to each channel
        slic = [
            self._apply_slic(np_image[i, ..., np.newaxis])
            for i in range(image.shape[0])
        ]
        if self.return_labels:
            slic_labels = np.transpose(np.array(slic), (1, 2, 0))
            slic_labels = self.tensor_transform(slic_labels)
            if self.labels_new_dimension:
                slic_labels = torch.stack((image, slic_labels))
            return slic_labels
        return slic

    def _apply_slic(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Applies SLIC to the input numpy array image."""
        # If gaussian_blur is not None, apply Gaussian blur to the image
        if self.gaussian_blur is not None:
            image = cv2.GaussianBlur(
                src=image,
                ksize=self.gaussian_blur,
                sigmaX=self.sigma[0],
                sigmaY=self.sigma[1],
            )
        slic_object = cv2.ximgproc.createSuperpixelSLIC(
            image, self.slic_algorithm, self.region_size, self.ruler
        )
        slic_object.iterate(self.slic_iterations)
        if self.enforce_connectivity:
            slic_object.enforceLabelConnectivity(self.min_element_size)
        # Get superpixel mask
        if self.return_labels:
            return slic_object.getLabels()
        return slic_object

    def _validate_args(self) -> None:
        """Validate the arguments passed to the class."""
        # Validate return_mask and return_segmented_image
        if not isinstance(self.return_labels, bool):
            raise TypeError(
                f"Invalid type for return_mask: {type(self.return_labels)}. "
                f"Expected bool."
            )
        if not isinstance(self.return_slic_object, bool):
            raise TypeError(
                f"Invalid type for return_segmented_image: {type(self.return_slic_object)}. "
                f"Expected bool."
            )
        if self.return_labels and self.return_slic_object:
            raise ValueError(
                "return_mask and return_segmented_image cannot both be True."
            )
        if not self.return_labels and not self.return_slic_object:
            raise ValueError(
                "At least one of return_mask and return_segmented_image must be True."
            )


class SegmentImageLogTale(torch.nn.Module):
    """A PyTorch module that applies SLIC and DBSCAN to an image.

    The method is applied individually to each channel of the input image.

    The method first applies SLIC to the input image to generate superpixels. Then,
    it applies DBSCAN to the superpixels to segment the image.

    The method can return the binary mask, the labels, or the segmented image.
    """

    def __init__(
        self,
        project_path,
        labels_new_dimension: bool = False,
        return_segmented_image: bool = False,
    ) -> None:
        """Initialize the SLICDBSCANSegmentation class.

        Args:
            labels_new_dimension (bool): If True, add a new dimension to the labels.
                Ignored if return_mask is False.
                Default is False.
            return_segmented_image (bool): If True, return the segmented image.
                Mutually exclusive with return_mask.
                At least one of return_mask and return_segmented_image must be True.
                Default is False.
        """
        super().__init__()

        self.slic_algorithm = 100
        self.region_size = 9
        self.ruler = 10.129958774739128
        self.enforce_connectivity = True
        self.min_element_size = 24
        self.min_label_size = 16
        self.eps = 5.397594617872189
        self.min_samples = 2
        self.leaf_size = 53
        self.p = 1.555065836800683
        self.slic_iterations = 100
        self.gaussian_blur = None
        self.sigma = (0.0, 0.0)
        self.dbscan_features = None
        self.metric = "euclidean"
        self.metric_params = None
        self.algorithm = "auto"
        self.use_radiomics = True
        self.dbscan_features = None
        self.use_pca = True
        self.n_jobs = -1
        self.return_mask = True

        self.labels_new_dimension: bool = labels_new_dimension
        self.return_segmented_image: bool = return_segmented_image
        self.pil_transform = T.ToPILImage()
        self.tensor_transform = T.ToImage()
        self._validate_args()

    def forward(self, image: Tensor) -> Tensor:
        """Applies SLIC and DBSCAN to the input tensor image.

        The method is applied individually to each channel of the input image.

        Args:
            image (torch.Tensor): The input tensor image.
                Shape should be (C, height, width).

        Returns:
            Tensor: The resulting tensor image mask.
                Shape will be (C, height, width).
        """
        # Instantiate SLICSegmentation and forward tensor image
        slic_segmentation = SLICSegmentation(
            slic_algorithm=self.slic_algorithm,
            region_size=self.region_size,
            ruler=self.ruler,
            slic_iterations=self.slic_iterations,
            gaussian_blur=self.gaussian_blur,
            sigma=self.sigma,
            enforce_connectivity=self.enforce_connectivity,
            min_element_size=self.min_element_size,
            return_labels=True,
            labels_new_dimension=False,
            return_slic_object=False,
        )
        slic_labels = slic_segmentation(image)
        # Convert tensor to numpy array for each channel
        np_image: np.ndarray = np.array(
            [
                self.pil_transform(image[i].unsqueeze(0)).convert("L")
                for i in range(image.shape[0])
            ]
        )
        np_slic_labels: np.ndarray = np.array(slic_labels)
        np_slic_labels = self._filter_label_size(np_slic_labels)
        # Apply DBSCAN to each channel
        slic_dbscan = []
        for i in range(image.shape[0]):
            # If labels.min() <= 0, increment labels until min is 1
            while np_slic_labels.min() <= 0:
                np_slic_labels += 1
            # Account for "empty" superpixels
            # If there is a missing label from 0 to max, decrement all labels greater than the missing label
            # Iteratively, until len(unique_slic_labels) == np_slic_labels.max()
            while len(np.unique(np_slic_labels)) < np_slic_labels.max():
                missing_label = np.sort(
                    np.setdiff1d(
                        np.arange(1, np_slic_labels.max() + 1),
                        np.unique(np_slic_labels),
                    )
                )[0]
                np_slic_labels[np_slic_labels > missing_label] -= 1
            superpixel_features = self._summarize_superpixels(
                np_image[i], np_slic_labels[i]
            )
            dbscan_labels = self._apply_dbscan(superpixel_features)
            # If labels.min() <= 0, increment labels until min is 1
            while dbscan_labels.min() <= 0:
                dbscan_labels += 1
            dbscan_labels = self._desummarize_superpixels(
                np_slic_labels[i], dbscan_labels
            )
            slic_dbscan.append(dbscan_labels)
        slic_dbscan_labels = np.transpose(np.array(slic_dbscan), (1, 2, 0)).astype(
            np.int32
        )
        slic_dbscan_labels = self.tensor_transform(slic_dbscan_labels)
        if self.return_mask:
            if self.labels_new_dimension:
                slic_dbscan_labels = torch.stack((image, slic_dbscan_labels.float()))
            return slic_dbscan_labels
        return slic_dbscan_labels * image

    def _filter_label_size(self, labels: np.ndarray) -> np.ndarray:
        """Filters the labels based on the minimum label size.

        Args:
            labels (np.ndarray): The input numpy array image labels.
                Shape should be (height, width).

        Returns:
            np.ndarray: The resulting numpy array image labels.
        """
        if self.min_label_size is None:
            return labels
        label_sizes = np.bincount(labels.flatten())
        mask = label_sizes[labels] >= self.min_label_size
        labels = mask * labels
        return labels

    def _summarize_superpixels(
        self, image: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Summarizes the superpixels and produces a feature matrix.

        Args:
            image (np.ndarray): The input numpy array image.
                Shape should be (height, width).
            labels (np.ndarray): The input numpy array image labels.
                Shape should be (height, width).

        Returns:
            np.ndarray: The resulting numpy array feature matrix.
        """
        if self.use_radiomics:
            return self._summarize_superpixels_radiomics(image, labels)
        return self._summarize_superpixels_skimage(image, labels)

    def _summarize_superpixels_skimage(
        self, image: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Summarizes the superpixels using skimage and produces a feature matrix.

        Args:
            image (np.ndarray): The input numpy array image.
                Shape should be (height, width).
            labels (np.ndarray): The input numpy array image labels.
                Shape should be (height, width).

        Returns:
            np.ndarray: The resulting numpy array feature matrix.
        """
        superpixel_features = []
        label_measures = measure.regionprops(
            label_image=labels,
            intensity_image=image,
            cache=True,
        )
        # TODO: Find errror in skimage (assertion)
        assert len(label_measures) == labels.max(), (
            f"_summarize_superpixels_skimage: len(label_measures)={len(label_measures)} "
            f"!= labels.max()={labels.max()}"
        )

        for measures in label_measures:
            row = []
            for feature in self.dbscan_features:
                # If tuple or ndarray, append each value of measures[feature]
                if isinstance(measures[feature], (tuple, list)):
                    row.extend(measures[feature])
                elif isinstance(measures[feature], np.ndarray):
                    row.extend(measures[feature].flatten().tolist())
                else:
                    row.append(measures[feature])
            superpixel_features.append(row)
        # Normalize the features
        superpixel_features = np.array(superpixel_features)
        superpixel_features = (
            superpixel_features - superpixel_features.mean(axis=0)
        ) / (superpixel_features.std(axis=0) + 1e-6)
        # If use_pca is True, apply PCA to the features
        if self.use_pca:
            # Estimate intrinsic dimensionality with broken_stick
            num_pca = broken_stick(superpixel_features, normalize=False)
            pca = PCA(n_components=num_pca)
            superpixel_features = pca.fit_transform(superpixel_features)

        return superpixel_features

    def _summarize_superpixels_radiomics(
        self, image: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Summarizes the superpixels using radiomics and produces a feature matrix.

        Args:
            image (np.ndarray): The input numpy array image.
                Shape should be (height, width).
            labels (np.ndarray): The input numpy array image labels.
                Shape should be (height, width).

        Returns:
            np.ndarray: The resulting numpy array feature matrix.
        """
        superpixel_features = []
        sitk_image = sitk.GetImageFromArray(image[..., np.newaxis])
        with ProcessPoolExecutor(
            None if self.n_jobs == -1 else self.n_jobs
        ) as executor:
            futures = []
            for label in np.unique(labels):
                mask = labels == label
                mask = mask.astype(int)[..., np.newaxis]
                mask = sitk.GetImageFromArray(mask)
                mask.CopyInformation(sitk_image)
                future = executor.submit(
                    self._extract_features_radiomics,
                    sitk_image,
                    mask,
                    self.dbscan_features,
                )
                futures.append(future)
            results = [future.result() for future in futures]
        superpixel_features = np.array(results)
        # Normalize the features
        superpixel_features = (
            superpixel_features - superpixel_features.mean(axis=0)
        ) / (superpixel_features.std(axis=0) + 1e-6)
        # If use_pca is True, apply PCA to the features
        if self.use_pca:
            # Estimate intrinsic dimensionality with broken_stick
            num_pca = broken_stick(superpixel_features, normalize=False)
            pca = PCA(n_components=num_pca)
            superpixel_features = pca.fit_transform(superpixel_features)

        return superpixel_features

    def _extract_features_radiomics(
        self,
        image: sitk.Image,
        mask: sitk.Image,
        dbscan_features: Optional[list[str]] = None,
    ) -> np.ndarray:
        """Extract features using radiomics.

        Args:
            image (sitk.Image): The input SimpleITK image.
            mask (sitk.Image): The input SimpleITK mask.
            dbscan_features (Optional[list[str]]): The features to use for DBSCAN.
                If None, all non-metadata features are used.
                Default is None.
        """
        feature_extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(
            normalize=True, force2D=True, force2Ddimension=2, minimumROIDimensions=1
        )
        feature_extractor.enableAllFeatures()
        result = feature_extractor.execute(image, mask)
        feature_set = np.array([])
        if dbscan_features is None:
            # If dbscan_features is None, use all non-metadata features
            for key, value in six.iteritems(result):
                if not key.startswith("diagnostics_"):
                    feature_set = np.append(feature_set, value)
        else:
            # Use only the specified features
            for feature in dbscan_features:
                feature_set = np.append(feature_set, result[feature])
        return feature_set

    def _desummarize_superpixels(
        self, slic_labels: np.ndarray, slic_dbscan_labels: np.ndarray
    ) -> np.ndarray:
        """Desummarizes the superpixels and produces a grouped superpixel mask.

        Args:
            slic_labels (np.ndarray): The input numpy array image labels.
                Shape should be (height, width).
            slic_dbscan_labels (np.ndarray): The input numpy array image labels.
                Shape should be (height, width).

        Returns:
            np.ndarray: The resulting numpy array grouped superpixel mask.
        """
        assert len(slic_dbscan_labels) == slic_labels.max(), (
            f"_desummarize_superpixels: len(slic_dbscan_labels)={len(slic_dbscan_labels)} "
            f"!= slic_labels.max()={slic_labels.max()}"
        )
        desummarized_mask = np.zeros_like(slic_labels)
        for i, slic_dbscan_label in enumerate(slic_dbscan_labels):
            mask = slic_labels == i + 1
            assert (
                mask.sum() > 0
            ), f"_desummarize_superpixels: mask.sum()={mask.sum()} <= 0"
            desummarized_mask[mask] = slic_dbscan_label
        return desummarized_mask

    def _apply_dbscan(self, features: np.ndarray) -> np.ndarray:
        """Applies DBSCAN to the input numpy array image.

        Args:
            features (np.ndarray): The input numpy array image features.
                Shape should be (n_samples, n_features).

        Returns:
            np.ndarray: The resulting numpy array image labels.
        """
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            metric_params=self.metric_params,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        dbscan.fit(features)
        labels = dbscan.labels_
        assert len(labels) == len(
            features
        ), f"_apply_dbscan: len(labels)={len(labels)} != len(features)={len(features)}"

        return labels

    def _validate_args(self) -> None:
        """Validate the arguments passed to the class."""
        # Validate return_mask and return_segmented_image
        if not isinstance(self.return_mask, bool):
            raise TypeError(
                f"Invalid type for return_mask: {type(self.return_mask)}. "
                f"Expected bool."
            )
        if not isinstance(self.return_segmented_image, bool):
            raise TypeError(
                f"Invalid type for return_segmented_image: {type(self.return_segmented_image)}. "
                f"Expected bool."
            )
        if self.return_mask and self.return_segmented_image:
            raise ValueError(
                "return_mask and return_segmented_image cannot both be True."
            )
        if not self.return_mask and not self.return_segmented_image:
            raise ValueError(
                "At least one of return_mask and return_segmented_image must be True."
            )
