import faiss
import numpy as np
import pytest

import stitch


class TestSplitImage:
    def test_exact_division_produces_correctly_sized_and_positioned_chunks(self):
        image = np.arange(4 * 6 * 3).reshape(4, 6, 3).astype(np.uint8)
        chunks = stitch.splitImage(image, chunk_width=2, chunk_height=2)

        coords = [(x, y) for x, y, _ in chunks]
        assert coords == [(0, 0), (2, 0), (4, 0), (0, 2), (2, 2), (4, 2)]
        for x, y, chunk in chunks:
            assert chunk.shape == (2, 2, 3)
            np.testing.assert_array_equal(chunk, image[y:y + 2, x:x + 2])

    def test_non_divisible_dimensions_produce_partial_trailing_chunks(self):
        image = np.arange(5 * 5 * 3).reshape(5, 5, 3).astype(np.uint8)
        chunks = stitch.splitImage(image, chunk_width=2, chunk_height=2)

        assert len(chunks) == 9  # starts at 0, 2, 4 on both axes

        by_coord = {(x, y): chunk for x, y, chunk in chunks}
        assert by_coord[(0, 0)].shape == (2, 2, 3)
        assert by_coord[(4, 0)].shape == (2, 1, 3)  # right edge: 1 column left
        assert by_coord[(0, 4)].shape == (1, 2, 3)  # bottom edge: 1 row left
        assert by_coord[(4, 4)].shape == (1, 1, 3)  # bottom-right corner

        for (x, y), chunk in by_coord.items():
            np.testing.assert_array_equal(chunk, image[y:y + 2, x:x + 2])

    def test_chunk_larger_than_image_returns_single_full_size_chunk(self):
        image = np.arange(3 * 3 * 3).reshape(3, 3, 3).astype(np.uint8)
        chunks = stitch.splitImage(image, chunk_width=10, chunk_height=10)

        assert len(chunks) == 1
        x, y, chunk = chunks[0]
        assert (x, y) == (0, 0)
        np.testing.assert_array_equal(chunk, image)


@pytest.fixture
def small_index():
    """A tiny FAISS index with 3 frames of 2 descriptors each, well separated."""
    vectors = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [10, 10, 10, 10],
        [10, 11, 10, 10],
        [20, 20, 20, 20],
        [20, 21, 20, 20],
    ], dtype="float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    frame_to_descriptor_indices = [0, 0, 1, 1, 2, 2]
    frame_ids = ["frame_a.png", "frame_b.png", "frame_c.png"]
    return index, frame_ids, frame_to_descriptor_indices, vectors


class TestMatchFeatures:
    def test_none_descriptors_returns_none(self, small_index):
        index, frame_ids, mapping, _ = small_index
        assert stitch.matchFeatures(None, index, frame_ids, mapping) == (None, None)

    def test_empty_descriptors_returns_none(self, small_index):
        index, frame_ids, mapping, _ = small_index
        empty = np.empty((0, 4), dtype="float32")
        assert stitch.matchFeatures(empty, index, frame_ids, mapping) == (None, None)

    def test_unanimous_vote_picks_matching_frame(self, small_index):
        index, frame_ids, mapping, vectors = small_index
        descriptors = vectors[2:4]  # both belong to frame_b

        frame, votes = stitch.matchFeatures(descriptors, index, frame_ids, mapping, top_k=1)

        assert frame == "frame_b.png"
        assert votes == 2

    def test_majority_vote_wins_over_minority(self, small_index):
        index, frame_ids, mapping, vectors = small_index
        descriptors = np.array([vectors[0], vectors[0], vectors[4]], dtype="float32")

        frame, votes = stitch.matchFeatures(descriptors, index, frame_ids, mapping, top_k=1)

        assert frame == "frame_a.png"
        assert votes == 2

    def test_descriptor_dimension_lower_than_index_is_zero_padded(self):
        vectors = np.array([[1, 2, 0, 0], [5, 6, 0, 0]], dtype="float32")
        index = faiss.IndexFlatL2(4)
        index.add(vectors)
        frame_ids = ["low_a.png", "low_b.png"]
        mapping = [0, 1]

        descriptors = np.array([[1, 2]], dtype="float32")  # 2-D query, 4-D index

        frame, votes = stitch.matchFeatures(descriptors, index, frame_ids, mapping, top_k=1)

        assert frame == "low_a.png"
        assert votes == 1

    def test_descriptor_dimension_higher_than_index_returns_none(self, small_index):
        index, frame_ids, mapping, _ = small_index
        descriptors = np.zeros((1, 6), dtype="float32")  # index expects 4-D

        assert stitch.matchFeatures(descriptors, index, frame_ids, mapping) == (None, None)

    def test_out_of_bounds_descriptor_index_falls_back(self):
        vectors = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype="float32")
        index = faiss.IndexFlatL2(4)
        index.add(vectors)
        frame_ids = ["fallback.png"]
        mapping = []  # every descriptor_index is out of bounds for this mapping

        frame, votes = stitch.matchFeatures(vectors.copy(), index, frame_ids, mapping, top_k=1)

        assert frame == "fallback.png"
        assert votes == 0

    def test_out_of_bounds_frame_index_falls_back(self):
        vectors = np.array([[0, 0, 0, 0]], dtype="float32")
        index = faiss.IndexFlatL2(4)
        index.add(vectors)
        frame_ids = ["only_frame.png"]
        mapping = [5]  # points past the end of frame_ids

        frame, votes = stitch.matchFeatures(vectors.copy(), index, frame_ids, mapping, top_k=1)

        assert frame == "only_frame.png"
        assert votes == 0


class TestFallbackFrame:
    def test_returns_first_frame_when_present(self):
        assert stitch.fallbackFrame(["a.png", "b.png"]) == "a.png"

    def test_returns_none_when_empty(self):
        assert stitch.fallbackFrame([]) is None
