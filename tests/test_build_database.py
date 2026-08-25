import cv2
import numpy as np
import pytest

import build_database


def _make_checkerboard(size=64, square=8):
    """A high-contrast synthetic image SIFT reliably finds keypoints in."""
    board = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, square):
        for j in range(0, size, square):
            if (i // square + j // square) % 2 == 0:
                board[i:i + square, j:j + square] = 255
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def _make_blank(size=64):
    """A featureless image: SIFT detects zero keypoints on this."""
    return np.zeros((size, size, 3), dtype=np.uint8)


class TestBuildDatabase:
    def test_indexes_descriptors_and_maps_them_back_to_their_frame(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        cv2.imwrite(str(input_dir / "frame0000.png"), _make_checkerboard())
        cv2.imwrite(str(input_dir / "frame0001.png"), _make_checkerboard(square=16))

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        faiss_index, frame_ids, frame_to_descriptor_indices = build_database.buildDatabase(
            input_dir, output_dir=output_dir
        )

        assert list(frame_ids) == ["frame0000.png", "frame0001.png"]
        assert faiss_index.ntotal == len(frame_to_descriptor_indices)
        assert faiss_index.d == build_database.DESCRIPTOR_DIM

        # Every descriptor's recorded frame index must resolve to a real frame.
        assert all(0 <= idx < len(frame_ids) for idx in frame_to_descriptor_indices)
        # And every frame must have contributed at least one descriptor.
        assert set(frame_to_descriptor_indices) == {0, 1}

        assert (output_dir / "individual_descriptors_faiss_index.bin").exists()
        assert (output_dir / "frame_ids.npy").exists()
        assert (output_dir / "frame_to_descriptor_indices.npy").exists()

    def test_skipped_frames_do_not_desync_the_descriptor_to_frame_mapping(self, tmp_path):
        """A frame with no detectable features (or a non-image file) must be
        skipped without leaving a gap between frame_ids and the frame indices
        recorded in frame_to_descriptor_indices."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        cv2.imwrite(str(input_dir / "frame0000.png"), _make_blank())  # no keypoints -> skipped
        cv2.imwrite(str(input_dir / "frame0001.png"), _make_checkerboard())
        (input_dir / "notes.txt").write_text("not an image")  # filtered out entirely
        cv2.imwrite(str(input_dir / "frame0002.png"), _make_checkerboard(square=16))

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _, frame_ids, frame_to_descriptor_indices = build_database.buildDatabase(
            input_dir, output_dir=output_dir
        )

        # Only the two frames with detectable features are indexed.
        assert list(frame_ids) == ["frame0001.png", "frame0002.png"]
        # Every recorded frame index must point at one of those two frames --
        # this fails if the blank/skipped frame still consumed an index slot.
        assert set(frame_to_descriptor_indices) <= {0, 1}
        assert set(frame_to_descriptor_indices) == {0, 1}

    def test_empty_folder_produces_an_empty_but_valid_index(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        faiss_index, frame_ids, frame_to_descriptor_indices = build_database.buildDatabase(
            input_dir, output_dir=output_dir
        )

        assert frame_ids == []
        assert frame_to_descriptor_indices == []
        assert faiss_index.ntotal == 0

    def test_unknown_index_type_raises(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        with pytest.raises(ValueError):
            build_database.buildDatabase(input_dir, output_dir=tmp_path, index_type="not_a_real_index_type")


class TestIndexTypes:
    """Both the incrementally-filled types (flat, hnsw) and the
    trained-then-bulk-filled types (ivfflat, ivfpq) must end up with the
    same descriptor-to-frame mapping guarantees."""

    @pytest.mark.parametrize("index_type", build_database.INDEX_TYPES)
    def test_every_index_type_produces_a_consistent_mapping(self, tmp_path, index_type):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        cv2.imwrite(str(input_dir / "frame0000.png"), _make_checkerboard())
        cv2.imwrite(str(input_dir / "frame0001.png"), _make_checkerboard(square=16))
        cv2.imwrite(str(input_dir / "frame0002.png"), _make_checkerboard(size=96))

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        faiss_index, frame_ids, frame_to_descriptor_indices = build_database.buildDatabase(
            input_dir, output_dir=output_dir, index_type=index_type
        )

        assert list(frame_ids) == ["frame0000.png", "frame0001.png", "frame0002.png"]
        assert faiss_index.ntotal == len(frame_to_descriptor_indices)
        assert faiss_index.d == build_database.DESCRIPTOR_DIM
        assert all(0 <= idx < len(frame_ids) for idx in frame_to_descriptor_indices)
        assert set(frame_to_descriptor_indices) == {0, 1, 2}

        if hasattr(faiss_index, "nprobe"):
            # IVF indexes default to nprobe=1 (one inverted-list cell per
            # query), which silently tanks recall unless raised explicitly.
            assert faiss_index.nprobe == build_database.IVF_NPROBE

    @pytest.mark.parametrize("index_type", build_database.TRAINED_INDEX_TYPES)
    def test_trained_index_types_handle_an_empty_folder_without_crashing(self, tmp_path, index_type):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        faiss_index, frame_ids, frame_to_descriptor_indices = build_database.buildDatabase(
            input_dir, output_dir=output_dir, index_type=index_type
        )

        assert frame_ids == []
        assert frame_to_descriptor_indices == []
        assert faiss_index.ntotal == 0


class TestGpuBuild:
    def test_use_gpu_with_unsupported_index_type_raises(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        with pytest.raises(ValueError):
            build_database.buildDatabase(input_dir, output_dir=tmp_path, index_type="hnsw", use_gpu=True)

    def test_use_gpu_without_a_gpu_raises(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        monkeypatch.setattr(build_database.faiss, "get_num_gpus", lambda: 0)

        with pytest.raises(RuntimeError):
            build_database.buildDatabase(input_dir, output_dir=tmp_path, index_type="ivfflat", use_gpu=True)

    @pytest.mark.skipif(build_database.faiss.get_num_gpus() == 0, reason="requires a Faiss-visible GPU")
    def test_use_gpu_produces_a_cpu_loadable_index_with_a_consistent_mapping(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        cv2.imwrite(str(input_dir / "frame0000.png"), _make_checkerboard())
        cv2.imwrite(str(input_dir / "frame0001.png"), _make_checkerboard(square=16))
        cv2.imwrite(str(input_dir / "frame0002.png"), _make_checkerboard(size=96))

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        faiss_index, frame_ids, frame_to_descriptor_indices = build_database.buildDatabase(
            input_dir, output_dir=output_dir, index_type="ivfflat", use_gpu=True
        )

        assert list(frame_ids) == ["frame0000.png", "frame0001.png", "frame0002.png"]
        assert faiss_index.ntotal == len(frame_to_descriptor_indices)
        assert faiss_index.nprobe == build_database.IVF_NPROBE
        # Not a faiss.GpuIndex -- must be loadable/queryable without a GPU.
        assert isinstance(faiss_index, build_database.faiss.IndexIVFFlat)

        # write_index/read_index must also round-trip a plain CPU index.
        index_path = output_dir / "individual_descriptors_faiss_index.bin"
        reloaded = build_database.faiss.read_index(str(index_path))
        assert reloaded.ntotal == faiss_index.ntotal
        assert reloaded.nprobe == build_database.IVF_NPROBE
