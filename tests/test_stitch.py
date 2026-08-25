import os

import cv2
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


class _FakeKeypoint:
    def __init__(self, x, y):
        self.pt = (x, y)


class _FakeDetector:
    """Stub for cv2.SIFT_create() with fixed keypoints/descriptors, so
    detectFeaturesForChunks' bucketing logic can be tested against known
    pixel locations without depending on what real SIFT happens to find."""

    def __init__(self, points_and_descriptors):
        self._keypoints = [_FakeKeypoint(x, y) for x, y, _ in points_and_descriptors]
        self._descriptors = (
            np.array([d for _, _, d in points_and_descriptors], dtype='float32')
            if points_and_descriptors else None
        )

    def detectAndCompute(self, image, mask):
        return self._keypoints, self._descriptors


def _checkerboard(size=64, square=8):
    board = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, square):
        for j in range(0, size, square):
            if (i // square + j // square) % 2 == 0:
                board[i:i + square, j:j + square] = 255
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


class TestDetectFeaturesForChunks:
    def test_buckets_keypoints_into_their_containing_chunk(self):
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        chunks = stitch.splitImage(image, chunk_width=2, chunk_height=2)  # 3x2 grid, 6 chunks

        detector = _FakeDetector([
            (0.5, 0.5, [1, 1]),  # chunk (0, 0) -> position 0
            (3.0, 0.0, [2, 2]),  # chunk (2, 0) -> position 1
            (5.9, 3.9, [3, 3]),  # chunk (4, 2) -> position 5
        ])

        result = stitch.detectFeaturesForChunks(image, chunks, chunk_width=2, chunk_height=2, detector=detector)

        assert len(result) == 6
        np.testing.assert_array_equal(result[0], [[1, 1]])
        np.testing.assert_array_equal(result[1], [[2, 2]])
        assert result[2] is None
        assert result[3] is None
        assert result[4] is None
        np.testing.assert_array_equal(result[5], [[3, 3]])

    def test_no_keypoints_returns_all_none(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        chunks = stitch.splitImage(image, chunk_width=2, chunk_height=2)
        detector = _FakeDetector([])

        result = stitch.detectFeaturesForChunks(image, chunks, chunk_width=2, chunk_height=2, detector=detector)

        assert result == [None] * len(chunks)

    def test_multiple_keypoints_in_the_same_chunk_are_grouped(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        chunks = stitch.splitImage(image, chunk_width=2, chunk_height=2)
        detector = _FakeDetector([(0.0, 0.0, [1, 0]), (1.5, 1.5, [0, 1])])

        result = stitch.detectFeaturesForChunks(image, chunks, chunk_width=2, chunk_height=2, detector=detector)

        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [[1, 0], [0, 1]])

    def test_keypoint_beyond_the_last_chunk_boundary_clamps_instead_of_dropping(self):
        # SIFT's sub-pixel keypoint refinement can place kp.pt fractionally
        # outside [0, image_size) at the edges; this must clamp into the
        # nearest real chunk rather than computing an out-of-range bucket.
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        chunks = stitch.splitImage(image, chunk_width=2, chunk_height=2)
        detector = _FakeDetector([(4.0, 4.0, [9, 9])])  # exactly at the image's far edge

        result = stitch.detectFeaturesForChunks(image, chunks, chunk_width=2, chunk_height=2, detector=detector)

        np.testing.assert_array_equal(result[3], [[9, 9]])  # bottom-right chunk

    def test_total_bucketed_descriptors_matches_whole_image_detection(self):
        # Sanity check against real SIFT (not the fake detector): every
        # descriptor detectAndCompute finds on the whole image must land in
        # exactly one chunk's bucket.
        board = _checkerboard(size=64, square=8)
        chunks = stitch.splitImage(board, chunk_width=16, chunk_height=16)
        detector = cv2.SIFT_create()

        result = stitch.detectFeaturesForChunks(board, chunks, chunk_width=16, chunk_height=16, detector=detector)

        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        _, whole_image_descriptors = detector.detectAndCompute(gray, None)

        total_bucketed = sum(len(bucket) for bucket in result if bucket is not None)
        assert total_bucketed == len(whole_image_descriptors)


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

    def test_search_returning_fewer_than_top_k_falls_back_instead_of_wrapping(self, small_index):
        # Faiss fills unfilled neighbor slots with -1 when a search finds
        # fewer than top_k candidates. Python indexing would silently
        # resolve mapping[-1] to the *last* entry instead of treating it as
        # invalid, so this must be handled explicitly rather than only
        # bounds-checking the upper end.
        index, frame_ids, mapping, vectors = small_index
        descriptors = vectors[0:1]
        top_k = len(vectors) + 5  # guarantees -1 padding in the result

        frame, votes = stitch.matchFeatures(descriptors, index, frame_ids, mapping, top_k=top_k)

        assert frame == "frame_a.png"
        assert votes == 2  # both frame_a descriptors are the nearest real neighbors


class TestMatchFeaturesBatch:
    def test_matches_each_chunk_independently_in_one_search_call(self, small_index, monkeypatch):
        index, frame_ids, mapping, vectors = small_index

        search_calls = []
        original_search = index.search

        def counting_search(*args, **kwargs):
            search_calls.append(1)
            return original_search(*args, **kwargs)

        monkeypatch.setattr(index, "search", counting_search)

        chunk_descriptors = [
            vectors[0:1],  # frame_a
            vectors[2:4],  # frame_b (both belong to it)
            vectors[4:5],  # frame_c
        ]

        results = stitch.matchFeaturesBatch(chunk_descriptors, index, frame_ids, mapping, top_k=1)

        assert len(search_calls) == 1  # one batched call, not one per chunk
        assert results == [
            ("frame_a.png", 1),
            ("frame_b.png", 2),
            ("frame_c.png", 1),
        ]

    def test_none_and_empty_chunks_yield_none_without_breaking_other_chunks(self, small_index):
        index, frame_ids, mapping, vectors = small_index
        chunk_descriptors = [None, vectors[0:1], np.empty((0, 4), dtype="float32")]

        results = stitch.matchFeaturesBatch(chunk_descriptors, index, frame_ids, mapping, top_k=1)

        assert results == [(None, None), ("frame_a.png", 1), (None, None)]

    def test_all_chunks_invalid_returns_all_none(self, small_index):
        index, frame_ids, mapping, _ = small_index
        chunk_descriptors = [None, np.empty((0, 4), dtype="float32")]

        results = stitch.matchFeaturesBatch(chunk_descriptors, index, frame_ids, mapping, top_k=1)

        assert results == [(None, None), (None, None)]

    def test_matches_matchFeatures_output_for_each_chunk(self, small_index):
        index, frame_ids, mapping, vectors = small_index
        chunk_descriptors = [vectors[0:1], vectors[2:4], vectors[4:5]]

        batch_results = stitch.matchFeaturesBatch(chunk_descriptors, index, frame_ids, mapping, top_k=1)
        individual_results = [
            stitch.matchFeatures(d, index, frame_ids, mapping, top_k=1) for d in chunk_descriptors
        ]

        assert batch_results == individual_results


class TestQuiltImage:
    def test_missing_frame_file_fills_black_instead_of_crashing(self, tmp_path):
        # cv2.imread returns None (not an exception) for a missing file, so
        # quiltImage must check for None itself rather than catching
        # FileNotFoundError, which cv2.imread never raises.
        chunk_results = [(0, 0, "does_not_exist.png", 5)]

        quilted = stitch.quiltImage(chunk_results, str(tmp_path), (2, 2, 3), chunk_width=2, chunk_height=2)

        np.testing.assert_array_equal(quilted, np.zeros((2, 2, 3), dtype=np.uint8))

    def test_none_match_is_skipped(self, tmp_path):
        chunk_results = [(0, 0, None, None)]

        quilted = stitch.quiltImage(chunk_results, str(tmp_path), (2, 2, 3), chunk_width=2, chunk_height=2)

        np.testing.assert_array_equal(quilted, np.zeros((2, 2, 3), dtype=np.uint8))

    def test_matched_frame_is_placed_at_its_chunk_position(self, tmp_path):
        frame = np.full((4, 4, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "frame.png"), frame)
        chunk_results = [(2, 0, "frame.png", 3)]

        quilted = stitch.quiltImage(chunk_results, str(tmp_path), (2, 4, 3), chunk_width=2, chunk_height=2)

        np.testing.assert_array_equal(quilted[0:2, 2:4], np.full((2, 2, 3), 200, dtype=np.uint8))
        np.testing.assert_array_equal(quilted[0:2, 0:2], np.zeros((2, 2, 3), dtype=np.uint8))

    def test_repeated_matches_read_the_source_frame_from_disk_only_once(self, tmp_path, monkeypatch):
        frame = np.full((4, 4, 3), 100, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "frame.png"), frame)

        read_calls = []
        original_imread = cv2.imread

        def counting_imread(path, *args, **kwargs):
            read_calls.append(path)
            return original_imread(path, *args, **kwargs)

        monkeypatch.setattr(cv2, "imread", counting_imread)

        chunk_results = [(0, 0, "frame.png", 1), (2, 0, "frame.png", 1), (0, 2, "frame.png", 1)]
        stitch.quiltImage(chunk_results, str(tmp_path), (4, 4, 3), chunk_width=2, chunk_height=2)

        assert len(read_calls) == 1


class TestFallbackFrame:
    def test_returns_first_frame_when_present(self):
        assert stitch.fallbackFrame(["a.png", "b.png"]) == "a.png"

    def test_returns_none_when_empty(self):
        assert stitch.fallbackFrame([]) is None


class TestProcessTargetImagesParallel:
    def test_parallel_output_matches_sequential_processing(self, tmp_path):
        import build_database  # only needed by this test

        mv_frames_dir = tmp_path / "mv_frames"
        mv_frames_dir.mkdir()
        for i in range(3):
            cv2.imwrite(str(mv_frames_dir / f"frame{i:04d}.png"), _checkerboard(size=64, square=8 + i * 4))

        db_dir = tmp_path / "db"
        db_dir.mkdir()
        build_database.buildDatabase(mv_frames_dir, output_dir=db_dir)

        target_dir = tmp_path / "targets"
        target_dir.mkdir()
        for i in range(4):
            cv2.imwrite(str(target_dir / f"target{i:04d}.png"), _checkerboard(size=64, square=8))

        faiss_index_path = db_dir / "individual_descriptors_faiss_index.bin"
        frame_ids_path = db_dir / "frame_ids.npy"
        descriptor_indices_path = db_dir / "frame_to_descriptor_indices.npy"

        # Both runs load the exact same on-disk index, so any index type is
        # equally deterministic here -- what's under test is that dividing
        # the frame list across worker processes doesn't change output.
        sequential_out = tmp_path / "sequential_out"
        sequential_out.mkdir()
        faiss_index = stitch.faiss.read_index(str(faiss_index_path))
        frame_ids = np.load(frame_ids_path)
        mapping = np.load(descriptor_indices_path)
        detector = cv2.SIFT_create()
        for i, filename in enumerate(stitch.list_image_files(str(target_dir)), start=1):
            stitch.processTargetImage(str(target_dir / filename), faiss_index, frame_ids, mapping,
                                       str(mv_frames_dir), i, str(sequential_out),
                                       chunk_width=16, chunk_height=16, detector=detector)

        parallel_out = tmp_path / "parallel_out"
        parallel_out.mkdir()
        count = stitch.processTargetImagesParallel(
            str(target_dir), faiss_index_path, frame_ids_path, descriptor_indices_path,
            str(mv_frames_dir), str(parallel_out), chunk_width=16, chunk_height=16, num_workers=2
        )

        assert count == 4
        sequential_files = sorted(os.listdir(sequential_out))
        parallel_files = sorted(os.listdir(parallel_out))
        assert sequential_files == parallel_files
        for filename in sequential_files:
            seq_img = cv2.imread(str(sequential_out / filename))
            par_img = cv2.imread(str(parallel_out / filename))
            np.testing.assert_array_equal(seq_img, par_img)

    def test_progress_callback_is_invoked_once_per_frame(self, tmp_path):
        import build_database  # only needed by this test

        mv_frames_dir = tmp_path / "mv_frames"
        mv_frames_dir.mkdir()
        cv2.imwrite(str(mv_frames_dir / "frame0000.png"), _checkerboard())

        db_dir = tmp_path / "db"
        db_dir.mkdir()
        build_database.buildDatabase(mv_frames_dir, output_dir=db_dir)

        target_dir = tmp_path / "targets"
        target_dir.mkdir()
        for i in range(3):
            cv2.imwrite(str(target_dir / f"target{i:04d}.png"), _checkerboard())

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        progress_calls = []
        stitch.processTargetImagesParallel(
            str(target_dir),
            db_dir / "individual_descriptors_faiss_index.bin",
            db_dir / "frame_ids.npy",
            db_dir / "frame_to_descriptor_indices.npy",
            str(mv_frames_dir), str(output_dir), chunk_width=16, chunk_height=16, num_workers=2,
            on_progress=progress_calls.append,
        )

        assert sorted(progress_calls) == [1, 2, 3]
