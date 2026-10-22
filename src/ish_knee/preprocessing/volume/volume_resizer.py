
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom


@dataclass(frozen=True)
class VolumeResizeConfig:
    target_shape: tuple[int, int, int] = (96, 160, 160)
    interpolation_order: int = 1


class VolumeResizer:
    """
    Resize a 3D MRI volume to a fixed shape.

    Axis convention:
        Axis 0 = depth / slice
        Axis 1 = row / height
        Axis 2 = column / width

    The input is expected to be a normalized 3D NumPy array.
    The output is a contiguous float32 NumPy array.
    """

    def __init__(
        self,
        config: VolumeResizeConfig | None = None,
    ) -> None:
        self.config = config or VolumeResizeConfig()

        if len(self.config.target_shape) != 3:
            raise ValueError("Target shape must contain three dimensions.")

        if any(
            not isinstance(size, int) or size <= 0
            for size in self.config.target_shape
        ):
            raise ValueError(
                "Target dimensions must be positive integers."
            )

        if self.config.interpolation_order not in (0, 1, 2, 3, 4, 5):
            raise ValueError(
                "Interpolation order must be between 0 and 5."
            )

    def resize(self, volume: np.ndarray) -> np.ndarray:
        """
        Resize a 3D volume to the configured target shape.

        Uses grid_mode=True so scaling follows the full voxel-grid
        extent. The returned shape is explicitly checked.
        """

        volume = np.asarray(volume)

        if volume.ndim != 3:
            raise ValueError(
                f"Expected a 3D volume, got {volume.ndim}D."
            )

        if volume.size == 0:
            raise ValueError("Volume cannot be empty.")

        if not np.issubdtype(volume.dtype, np.number):
            raise TypeError("Volume must contain numeric values.")

        if not np.isfinite(volume).all():
            raise ValueError(
                "Volume contains NaN or infinite values."
            )

        source = volume.astype(np.float32, copy=False)

        zoom_factors = tuple(
            target / original
            for target, original in zip(
                self.config.target_shape,
                source.shape,
            )
        )

        resized = zoom(
            source,
            zoom=zoom_factors,
            order=self.config.interpolation_order,
            mode="nearest",
            prefilter=self.config.interpolation_order > 1,
            grid_mode=True,
        )

        # Guard against unexpected library output dimensions.
        if resized.shape != self.config.target_shape:
            raise RuntimeError(
                f"Expected output shape {self.config.target_shape}, "
                f"received {resized.shape}."
            )

        if not np.isfinite(resized).all():
            raise RuntimeError(
                "Resized volume contains non-finite values."
            )

        return np.ascontiguousarray(resized, dtype=np.float32)
