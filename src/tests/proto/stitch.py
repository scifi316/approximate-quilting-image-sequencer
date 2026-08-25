import cv2
import numpy as np
import faiss
from pathlib import Path
import multiprocessing
import os
import sys
from collections import Counter

# Root directory solve
root_dir = Path(__file__).resolve().parents[3]  # src/tests/proto --> $main dir
sys.path.insert(0, str(root_dir))

from src.common import computeTileDescriptors, list_image_files  # noqa: E402


def splitImage(image, chunk_width, chunk_height):
    """Split an image into smaller chunks of the given size."""
    chunks = []
    h, w = image.shape[:2]

    for y in range(0, h, chunk_height):
        for x in range(0, w, chunk_width):
            chunk = image[y:y + chunk_height, x:x + chunk_width]
            chunks.append((x, y, chunk))

    return chunks


def detectFeaturesForChunks(image, chunks, chunk_width, chunk_height, detector=None):
    """Detect SIFT features once for the whole target image, then bucket
    each keypoint/descriptor into the chunk it falls in by pixel location.

    SIFT is used (rather than ORB) because the database built by
    build_database.py is indexed on 128-D SIFT descriptors; matching with a
    different descriptor type/dimensionality produces meaningless results.

    Running detectAndCompute once per frame instead of once per chunk (there
    are hundreds of chunks per frame) matters: SIFT pays fixed per-call
    overhead (building a scale-space pyramid, etc.) on every call regardless
    of how small the input is, so calling it per-chunk was the dominant cost
    of processing a frame -- roughly 2/3 of total per-frame time, well above
    the batched Faiss search itself. Pass a shared `detector` to also avoid
    the overhead of constructing a new cv2.SIFT instance per frame.

    Returns a list of descriptor arrays (or None where no keypoints fell in
    that chunk), aligned with `chunks` (as produced by splitImage).
    """
    detector = detector or cv2.SIFT_create()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray_image, None)

    buckets = [[] for _ in chunks]
    if not keypoints:
        return [None] * len(chunks)

    h, w = image.shape[:2]
    num_cols = -(-w // chunk_width)  # ceil division, matches splitImage's chunk grid
    num_rows = -(-h // chunk_height)

    for keypoint, descriptor in zip(keypoints, descriptors):
        x, y = keypoint.pt
        col = min(max(int(x) // chunk_width, 0), num_cols - 1)
        row = min(max(int(y) // chunk_height, 0), num_rows - 1)
        buckets[row * num_cols + col].append(descriptor)

    return [np.array(bucket, dtype='float32') if bucket else None for bucket in buckets]


def _padOrValidateDescriptors(descriptors, faiss_index):
    """Zero-pad descriptors up to the index's dimensionality if they're
    smaller, or return None if the dimensionality can't be reconciled."""
    descriptor_dim = descriptors.shape[1]
    if descriptor_dim < faiss_index.d:
        padded_descriptors = np.zeros((descriptors.shape[0], faiss_index.d), dtype='float32')
        padded_descriptors[:, :descriptor_dim] = descriptors
        return padded_descriptors
    if descriptor_dim != faiss_index.d:
        print(f"Error: Descriptor dimension {descriptor_dim} does not match Faiss index dimension {faiss_index.d}.")
        return None
    return descriptors


def _votesFromIndices(indices, frame_ids, frame_to_descriptor_indices):
    """Turn a batch of Faiss neighbor indices into a flat list of frame-ID votes."""
    votes = []
    for row in indices:
        for descriptor_index in row:
            # Faiss returns -1 for unfilled neighbor slots when a search finds
            # fewer than top_k candidates (routine with approximate indexes
            # such as IVF/HNSW). Without this check, -1 would silently wrap
            # around to frame_to_descriptor_indices[-1] via Python indexing.
            if descriptor_index < 0 or descriptor_index >= len(frame_to_descriptor_indices):
                if descriptor_index >= len(frame_to_descriptor_indices):
                    print(f"Error: descriptor_index {descriptor_index} is out of bounds for "
                          f"frame_to_descriptor_indices of length {len(frame_to_descriptor_indices)}.")
                continue

            frame_index = frame_to_descriptor_indices[descriptor_index]

            if frame_index >= len(frame_ids):
                continue

            votes.append(frame_ids[frame_index])
    return votes


# Match the detected features to the individual descriptors in the Faiss index using a voting mechanism
def matchFeatures(descriptors, faiss_index, frame_ids, frame_to_descriptor_indices, top_k=7):
    """Match the descriptors of a single chunk to the Faiss index using a voting mechanism."""
    if descriptors is None or len(descriptors) == 0:
        return None, None  # No features detected

    descriptors = _padOrValidateDescriptors(descriptors, faiss_index)
    if descriptors is None:
        return None, None

    _, indices = faiss_index.search(descriptors, top_k)
    votes = _votesFromIndices(indices, frame_ids, frame_to_descriptor_indices)

    if votes:
        return Counter(votes).most_common(1)[0]

    print("Warning: No valid matches found, using fallback mechanism.")
    return fallbackFrame(frame_ids), 0


def matchFeaturesBatch(chunk_descriptors, faiss_index, frame_ids, frame_to_descriptor_indices, top_k=7):
    """Batched version of matchFeatures: matches many chunks' descriptors with
    a single Faiss search call instead of one call per chunk. This is the
    matching hot path used by processTargetImage -- for a target image split
    into hundreds of chunks, one batched search is far cheaper than hundreds
    of individual ones.

    Returns a list of (frame_id, vote_count) tuples, one per entry in
    chunk_descriptors, in the same order.
    """
    results = [(None, None)] * len(chunk_descriptors)

    prepared_descriptors, chunk_positions, chunk_sizes = [], [], []
    for position, descriptors in enumerate(chunk_descriptors):
        if descriptors is None or len(descriptors) == 0:
            continue

        prepared = _padOrValidateDescriptors(descriptors, faiss_index)
        if prepared is None:
            continue

        prepared_descriptors.append(prepared)
        chunk_positions.append(position)
        chunk_sizes.append(prepared.shape[0])

    if not prepared_descriptors:
        return results

    all_descriptors = np.vstack(prepared_descriptors).astype('float32')
    _, indices = faiss_index.search(all_descriptors, top_k)

    offset = 0
    for position, size in zip(chunk_positions, chunk_sizes):
        chunk_indices = indices[offset:offset + size]
        offset += size

        votes = _votesFromIndices(chunk_indices, frame_ids, frame_to_descriptor_indices)
        if votes:
            results[position] = Counter(votes).most_common(1)[0]
        else:
            results[position] = (fallbackFrame(frame_ids), 0)

    return results


# Fallback mechanism will select the first frame ID, typically a black frame
def fallbackFrame(frame_ids):
    """Fallback mechanism to select a default frame if no valid match is found."""
    return frame_ids[0] if frame_ids else None


def quiltImage(chunk_results, mv_frames_folder, target_image_shape, chunk_width, chunk_height):
    """Quilt together the best-matching frames into the target image's shape."""
    h, w = target_image_shape[:2]
    quilted_image = np.zeros((h, w, 3), dtype=np.uint8)

    # Many chunks in the same target image often match the same source
    # frame; cache each frame's resized version instead of re-reading and
    # re-resizing it from disk on every chunk.
    resized_frame_cache = {}

    for x, y, match_frame_id, _ in chunk_results:
        if match_frame_id is None:
            # Skip this chunk if no valid match was found
            continue

        if match_frame_id not in resized_frame_cache:
            frame_path = os.path.join(mv_frames_folder, match_frame_id)
            mv_frame = cv2.imread(frame_path)
            if mv_frame is None:
                print(f"Frame {frame_path} not found.")
                resized_frame_cache[match_frame_id] = None
            else:
                resized_frame_cache[match_frame_id] = cv2.resize(mv_frame, (chunk_width, chunk_height))

        resized_frame = resized_frame_cache[match_frame_id]
        if resized_frame is None:
            # Fill in a default color for missing frames
            quilted_image[y:y + chunk_height, x:x + chunk_width] = (0, 0, 0)
        else:
            quilted_image[y:y + chunk_height, x:x + chunk_width] = resized_frame

    return quilted_image


def quiltImageGrid(chunk_results, mv_frames_folder, target_image_shape, chunk_width, chunk_height,
                    thumbnail_cache=None):
    """Vectorized equivalent of quiltImage for a regular, exact chunk grid
    (target_image_shape's width/height evenly divisible by chunk_width/
    chunk_height -- true for tile-descriptor mode by construction, since
    computeTileDescriptors requires exact division; not for SIFT mode's
    possible partial edge chunks, so use quiltImage there instead).

    quiltImage's per-chunk Python loop becomes the dominant per-frame cost
    once chunk counts get large (fine-grained quilting): the pixel-copy
    work itself is trivial, but thousands of individual dict lookups and
    numpy slice-assignments in a Python loop is not. This does the same
    work as a handful of vectorized numpy operations instead: resize each
    *unique* matched frame once, then gather + one reshape/transpose to
    assemble the whole canvas in a single shot -- no per-chunk Python loop.

    thumbnail_cache, if given, is a dict reused *across calls* (across
    target frames), mapping frame_id -> its resized thumbnail. The source
    database only has as many distinct frames as it was built from (~5K in
    this project's dataset), and their resized thumbnails are tiny, so
    caching across the whole run instead of re-reading+resizing a matched
    frame from disk every time it's matched again (common: many chunks per
    frame, across many frames, repeatedly match the same popular source
    frames) turns most quilting into a pure in-memory gather.
    """
    h, w = target_image_shape[:2]
    if w % chunk_width != 0 or h % chunk_height != 0:
        raise ValueError(f"Target size {w}x{h} isn't evenly divisible by chunk size {chunk_width}x{chunk_height}.")
    num_cols = w // chunk_width
    num_rows = h // chunk_height

    black_tile = np.zeros((chunk_height, chunk_width, 3), dtype=np.uint8)
    resized_by_frame_id = {None: black_tile} if thumbnail_cache is None else thumbnail_cache
    resized_by_frame_id.setdefault(None, black_tile)

    def _resizedTile(match_frame_id):
        if match_frame_id not in resized_by_frame_id:
            mv_frame = cv2.imread(os.path.join(mv_frames_folder, match_frame_id))
            if mv_frame is None:
                print(f"Frame {os.path.join(mv_frames_folder, match_frame_id)} not found.")
                resized_by_frame_id[match_frame_id] = black_tile
            else:
                resized_by_frame_id[match_frame_id] = cv2.resize(mv_frame, (chunk_width, chunk_height))
        return resized_by_frame_id[match_frame_id]

    match_frame_ids = [match_frame_id for _, _, match_frame_id, _ in chunk_results]
    # Only this call's distinct matches -- resized_by_frame_id may be a
    # cache shared across many calls and grow much larger than what's
    # needed here; stacking the whole cache every call would make this
    # scale with total frames seen so far instead of this frame's chunks.
    unique_ids = list(dict.fromkeys(match_frame_ids))  # de-duplicate, preserve order
    for match_frame_id in unique_ids:
        _resizedTile(match_frame_id)  # populate resized_by_frame_id for every distinct match, once each

    id_to_unique_index = {frame_id: index for index, frame_id in enumerate(unique_ids)}
    stacked_tiles = np.stack([resized_by_frame_id[frame_id] for frame_id in unique_ids])  # (U, ch, cw, 3)

    chunk_to_unique_index = np.array([id_to_unique_index[frame_id] for frame_id in match_frame_ids])
    all_tiles = stacked_tiles[chunk_to_unique_index]  # (num_chunks, chunk_height, chunk_width, 3)

    quilted_image = (
        all_tiles.reshape(num_rows, num_cols, chunk_height, chunk_width, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(h, w, 3)
    )
    return np.ascontiguousarray(quilted_image)


def _descriptorsForChunks(target_image, image_chunks, chunk_width, chunk_height, descriptor_type,
                           detector, thumb_size):
    """Dispatch to the configured descriptor extraction strategy. Must
    produce the same result shape either way: a list of descriptor arrays
    (or None), one per entry in image_chunks, in the same order."""
    if descriptor_type == "sift":
        return detectFeaturesForChunks(target_image, image_chunks, chunk_width, chunk_height, detector)

    if descriptor_type == "tile":
        # One dense descriptor per tile -- unlike SIFT, this never leaves a
        # chunk with nothing to match on, which is what makes small chunk
        # sizes (fine-grained quilting) viable. See
        # src.common.computeTileDescriptors and benchmarks/RESULTS.md.
        descriptors, _, _ = computeTileDescriptors(target_image, chunk_width, chunk_height, thumb_size)
        return [row.reshape(1, -1) for row in descriptors]

    raise ValueError(f"Unknown descriptor type {descriptor_type!r}; choose from 'sift' or 'tile'.")


def processTargetImage(target_image_path, faiss_index, frame_ids, frame_to_descriptor_indices,
                        mv_frames_folder, output_index, output_dir, chunk_width=96, chunk_height=72,
                        detector=None, descriptor_type="sift", thumb_size=4, thumbnail_cache=None,
                        upscale_to=None):
    """Process a single target image against an already-loaded Faiss index.

    faiss_index/frame_ids/frame_to_descriptor_indices are loaded once by the
    caller and reused across every target frame -- re-reading a
    multi-hundred-MB index file per frame made processing a whole video
    prohibitively slow. faiss_index may be CPU- or GPU-resident (Faiss's
    search() API is the same either way); GPU placement is the caller's
    responsibility (see _moveIndexToGpu), done once rather than per frame.

    descriptor_type must match how the database faiss_index was built with
    (build_database.buildDatabase's descriptor_type) -- "sift" and "tile"
    index different, incompatible vector spaces, and chunk_width/
    chunk_height/thumb_size must match what was used to build a "tile"
    database, since tile descriptors are tied to a specific grid alignment.

    upscale_to, if given, is an (width, height) pair the target image is
    resized to before chunking -- e.g. (3840, 2160) to chunk a 1920x1080
    source at 4K. This doesn't add real detail (it's the same source
    footage, just resampled larger), but it does pack more chunk_width x
    chunk_height tiles into the same visual content, since each tile then
    covers a proportionally smaller fraction of it -- a way to increase
    mosaic granularity without shrinking the tile pixel size itself. Must
    be evenly divisible by chunk_width/chunk_height for "tile" mode (see
    computeTileDescriptors).
    """
    # Load the target image
    target_image = cv2.imread(str(target_image_path))
    if target_image is None:
        raise FileNotFoundError(f"Target image {target_image_path} not found.")

    if upscale_to is not None:
        target_image = cv2.resize(target_image, upscale_to, interpolation=cv2.INTER_CUBIC)

    # Split the target image into chunks
    image_chunks = splitImage(target_image, chunk_width, chunk_height)
    detector = detector or cv2.SIFT_create()

    # Extract descriptors once for the whole frame, one entry per chunk,
    # then match all chunks in a single batched Faiss search
    chunk_descriptors = _descriptorsForChunks(target_image, image_chunks, chunk_width, chunk_height,
                                               descriptor_type, detector, thumb_size)
    matches = matchFeaturesBatch(chunk_descriptors, faiss_index, frame_ids, frame_to_descriptor_indices)

    chunk_results = [
        (x, y, match_frame_id, match_score)
        for (x, y, _), (match_frame_id, match_score) in zip(image_chunks, matches)
    ]

    # Quilt the final image using the matched frames. Tile mode guarantees
    # an exact chunk grid (computeTileDescriptors requires it), so it can
    # use quiltImageGrid's vectorized assembly; SIFT mode may have partial
    # edge chunks (image dims not evenly divisible by chunk size), which
    # only the general per-chunk quiltImage handles.
    if descriptor_type == "tile":
        quilted_image = quiltImageGrid(chunk_results, mv_frames_folder, target_image.shape, chunk_width,
                                        chunk_height, thumbnail_cache=thumbnail_cache)
    else:
        quilted_image = quiltImage(chunk_results, mv_frames_folder, target_image.shape, chunk_width, chunk_height)

    # Save the quilted image
    output_image_path = Path(output_dir) / f"quilted_frame{output_index:04d}.png"
    cv2.imwrite(str(output_image_path), quilted_image)


GPU_TEMP_MEMORY_BYTES = 128 * 1024 * 1024  # 128MB
GPU_PINNED_MEMORY_BYTES = 32 * 1024 * 1024  # 32MB


def _moveIndexToGpu(faiss_index):
    """Move a CPU Faiss index onto the GPU for searching. Returns
    (gpu_index, gpu_resources); gpu_resources must be kept alive by the
    caller for as long as gpu_index is used -- freeing it invalidates the
    index. Faiss's GPU backend doesn't support HNSW indexes.

    Caps each GpuResources' scratch buffer well below Faiss's ~1.5GB
    default temp memory, and its separate ~256MB default pinned
    CPU<->GPU async copy buffer: this pipeline runs many worker processes
    that each grab their own GpuResources sharing one GPU (see
    processTargetImagesParallel). Many workers each requesting the default
    256MB *pinned* (page-locked) allocation concurrently at pool startup
    intermittently failed with cudaHostAlloc errors ("resource already
    mapped" / device busy) under real contention -- pinned memory is a
    scarcer, more contended OS resource than regular heap, and this
    workload's search batches (a few MB at most) don't need anywhere near
    the default sizes.
    """
    if isinstance(faiss_index, faiss.IndexHNSWFlat):
        raise ValueError("HNSW indexes aren't supported on Faiss's GPU backend; build with flat or ivfflat instead.")
    gpu_resources = faiss.StandardGpuResources()
    gpu_resources.setTempMemory(GPU_TEMP_MEMORY_BYTES)
    gpu_resources.setPinnedMemory(GPU_PINNED_MEMORY_BYTES)
    gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, faiss_index)
    return gpu_index, gpu_resources


_worker_state = {}


def _initWorker(faiss_index_path, frame_ids_path, descriptor_indices_path, mv_frames_folder,
                 chunk_width, chunk_height, threads_per_worker, descriptor_type, thumb_size, use_gpu,
                 upscale_to):
    """multiprocessing.Pool initializer: each worker loads its own copy of
    the Faiss index once (Faiss indexes aren't picklable/shareable across
    process boundaries) and caps its own thread pools -- otherwise every
    worker process would try to claim all available cores for its own
    Faiss search and SIFT calls, oversubscribing the machine instead of
    dividing it up across workers.

    use_gpu moves each worker's own index copy onto the GPU once here
    (not per frame) -- multiple worker processes then share the one GPU for
    searching, same as they already share CPU cores for everything else.
    """
    faiss.omp_set_num_threads(threads_per_worker)
    cv2.setNumThreads(1)

    faiss_index = faiss.read_index(str(faiss_index_path))
    if use_gpu:
        faiss_index, gpu_resources = _moveIndexToGpu(faiss_index)
        _worker_state["gpu_resources"] = gpu_resources  # keep alive

    _worker_state["faiss_index"] = faiss_index
    _worker_state["frame_ids"] = np.load(frame_ids_path)
    _worker_state["frame_to_descriptor_indices"] = np.load(descriptor_indices_path)
    _worker_state["mv_frames_folder"] = mv_frames_folder
    _worker_state["chunk_width"] = chunk_width
    _worker_state["chunk_height"] = chunk_height
    _worker_state["descriptor_type"] = descriptor_type
    _worker_state["thumb_size"] = thumb_size
    _worker_state["upscale_to"] = upscale_to
    _worker_state["detector"] = cv2.SIFT_create() if descriptor_type == "sift" else None
    # Persists across every frame this worker processes (not per-frame): the
    # source database only has as many distinct frames as it was built
    # from, and their resized thumbnails are tiny, so this turns repeated
    # matches to popular source frames into in-memory gathers instead of
    # re-reading+resizing them from disk on every target frame that matches
    # them. Only used by quiltImageGrid (descriptor_type="tile").
    _worker_state["thumbnail_cache"] = {}


def _processOneFrameInWorker(args):
    target_image_path, output_index, output_dir = args
    processTargetImage(
        target_image_path,
        _worker_state["faiss_index"],
        _worker_state["frame_ids"],
        _worker_state["frame_to_descriptor_indices"],
        _worker_state["mv_frames_folder"],
        output_index,
        output_dir,
        chunk_width=_worker_state["chunk_width"],
        chunk_height=_worker_state["chunk_height"],
        detector=_worker_state["detector"],
        descriptor_type=_worker_state["descriptor_type"],
        thumb_size=_worker_state["thumb_size"],
        thumbnail_cache=_worker_state["thumbnail_cache"],
        upscale_to=_worker_state["upscale_to"],
    )
    return output_index


def processTargetImagesParallel(target_image_dir, faiss_index_path, frame_ids_path, descriptor_indices_path,
                                 mv_frames_folder, output_dir, chunk_width=96, chunk_height=72,
                                 num_workers=None, on_progress=None, descriptor_type="sift", thumb_size=4,
                                 use_gpu=False, upscale_to=None):
    """Process every target image in target_image_dir across multiple
    worker processes instead of one frame at a time in the calling process.

    Frames are independent -- each reads its own target image and writes
    its own output file -- so this parallelizes across cores in a way no
    further per-frame optimization can. num_workers defaults to
    cpu_count() - 1. on_progress, if given, is called with the number of
    frames completed so far after each one finishes (order not guaranteed
    to match frame order).

    descriptor_type/chunk_width/chunk_height/thumb_size must match how the
    database at faiss_index_path was built (build_database.buildDatabase).
    use_gpu moves each worker's own Faiss index copy onto the GPU for
    searching (not supported for HNSW-built indexes). upscale_to is an
    (width, height) pair each target frame is resized to before chunking --
    see processTargetImage's docstring.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_workers = max(1, num_workers or (multiprocessing.cpu_count() - 1))
    # Leave each worker a small thread budget of its own for Faiss's
    # OMP-parallel search, rather than pinning every worker to a single
    # thread and fully serializing that part of the work.
    threads_per_worker = max(1, multiprocessing.cpu_count() // num_workers)

    filenames = list_image_files(target_image_dir)
    tasks = [
        (os.path.join(target_image_dir, filename), output_index, output_dir)
        for output_index, filename in enumerate(filenames, start=1)
    ]

    with multiprocessing.Pool(
        processes=num_workers,
        initializer=_initWorker,
        initargs=(faiss_index_path, frame_ids_path, descriptor_indices_path, mv_frames_folder,
                  chunk_width, chunk_height, threads_per_worker, descriptor_type, thumb_size, use_gpu,
                  upscale_to),
    ) as pool:
        for completed, _ in enumerate(pool.imap_unordered(_processOneFrameInWorker, tasks), start=1):
            if on_progress:
                on_progress(completed)

    return len(tasks)


if __name__ == "__main__":
    # Define the paths to the target image, Faiss index, frame IDs, and MV frames
    target_image_path = root_dir / 'data/images/source'
    faiss_index_path = root_dir / 'individual_descriptors_faiss_index.bin'
    frame_ids_path = root_dir / 'frame_ids.npy'
    mv_frames_folder = root_dir / 'data/images/input'
    descriptor_indices_path = root_dir / 'frame_to_descriptor_indices.npy'
    output_dir = root_dir / 'data/images/quilted_output'

    # Frames are independent (each reads its own target image, writes its
    # own output file), so this parallelizes across processes rather than
    # processing one frame at a time. Override worker count with
    # QUILT_WORKERS=N; QUILT_WORKERS=1 falls back to effectively sequential.
    num_workers = int(os.environ.get('QUILT_WORKERS', 0)) or None
    # Must match the database's build_database.py descriptor_type/chunk
    # size/thumb_size (QUILT_DESCRIPTOR_TYPE/QUILT_CHUNK_WIDTH/
    # QUILT_CHUNK_HEIGHT/QUILT_THUMB_SIZE there) -- these are two ends of
    # the same descriptor space and will silently produce nonsense matches
    # if mismatched.
    descriptor_type = os.environ.get('QUILT_DESCRIPTOR_TYPE', 'sift')
    chunk_width = int(os.environ.get('QUILT_CHUNK_WIDTH', 96))
    chunk_height = int(os.environ.get('QUILT_CHUNK_HEIGHT', 72))
    thumb_size = int(os.environ.get('QUILT_THUMB_SIZE', 4))
    use_gpu = os.environ.get('QUILT_USE_GPU', '').lower() in ('1', 'true', 'yes')
    # Resize each target frame to QUILT_UPSCALE_WIDTHxQUILT_UPSCALE_HEIGHT
    # before chunking -- e.g. 3840x2160 to chunk a 1080p source at 4K,
    # packing more chunk_width x chunk_height tiles into the same footage
    # without shrinking the tile pixel size. See processTargetImage's
    # upscale_to docstring. Both must be set together.
    upscale_width = os.environ.get('QUILT_UPSCALE_WIDTH')
    upscale_height = os.environ.get('QUILT_UPSCALE_HEIGHT')
    upscale_to = (int(upscale_width), int(upscale_height)) if upscale_width and upscale_height else None
    total_frames = len(list_image_files(target_image_path))

    def _report_progress(completed):
        if completed % 100 == 0 or completed == total_frames:
            print(f"  {completed}/{total_frames} frames quilted")

    processTargetImagesParallel(target_image_path, faiss_index_path, frame_ids_path, descriptor_indices_path,
                                 mv_frames_folder, output_dir, chunk_width=chunk_width, chunk_height=chunk_height,
                                 num_workers=num_workers, on_progress=_report_progress,
                                 descriptor_type=descriptor_type, thumb_size=thumb_size, use_gpu=use_gpu,
                                 upscale_to=upscale_to)

    print("Quilted all images")
