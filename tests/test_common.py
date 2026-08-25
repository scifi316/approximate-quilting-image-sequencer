import numpy as np
import pytest

from src.common import computeTileDescriptors, tileDescriptorDim


class TestComputeTileDescriptors:
    def test_solid_color_tiles_produce_exact_per_tile_mean(self):
        # 2x2 grid of 2x2 tiles, each tile a distinct solid color. With
        # thumb_size=1, the descriptor for a solid tile must be exactly
        # that color (the area-average of a uniform region is itself).
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[0:2, 0:2] = (10, 20, 30)    # top-left
        image[0:2, 2:4] = (40, 50, 60)    # top-right
        image[2:4, 0:2] = (70, 80, 90)    # bottom-left
        image[2:4, 2:4] = (100, 110, 120)  # bottom-right

        descriptors, num_cols, num_rows = computeTileDescriptors(image, chunk_width=2, chunk_height=2, thumb_size=1)

        assert num_cols == 2
        assert num_rows == 2
        assert descriptors.shape == (4, tileDescriptorDim(1, channels=3))
        # Row-major order (y outer, x inner), matching splitImage's chunk grid.
        np.testing.assert_allclose(descriptors[0], [10, 20, 30], atol=1)
        np.testing.assert_allclose(descriptors[1], [40, 50, 60], atol=1)
        np.testing.assert_allclose(descriptors[2], [70, 80, 90], atol=1)
        np.testing.assert_allclose(descriptors[3], [100, 110, 120], atol=1)

    def test_descriptor_dtype_is_float32(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        descriptors, _, _ = computeTileDescriptors(image, chunk_width=2, chunk_height=2, thumb_size=1)
        assert descriptors.dtype == np.float32

    def test_thumb_size_scales_descriptor_dimension(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        descriptors, num_cols, num_rows = computeTileDescriptors(image, chunk_width=4, chunk_height=4, thumb_size=3)

        assert num_cols == 2
        assert num_rows == 2
        assert descriptors.shape == (4, 3 * 3 * 3)

    def test_non_dividing_chunk_size_raises(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            computeTileDescriptors(image, chunk_width=3, chunk_height=3, thumb_size=2)

    def test_grayscale_image_produces_single_channel_descriptors(self):
        image = np.zeros((4, 4), dtype=np.uint8)
        image[0:2, 0:2] = 50
        image[0:2, 2:4] = 150

        descriptors, num_cols, num_rows = computeTileDescriptors(image, chunk_width=2, chunk_height=2, thumb_size=1)

        assert descriptors.shape == (4, tileDescriptorDim(1, channels=1))
        np.testing.assert_allclose(descriptors[0], [50], atol=1)
        np.testing.assert_allclose(descriptors[1], [150], atol=1)


class TestTileDescriptorDim:
    def test_matches_thumb_size_squared_times_channels(self):
        assert tileDescriptorDim(4, channels=3) == 48
        assert tileDescriptorDim(1, channels=3) == 3
        assert tileDescriptorDim(4, channels=1) == 16
