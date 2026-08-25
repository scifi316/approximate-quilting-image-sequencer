# Installation
This is for installing the python 3.12 version of the program, please make sure all the requirements are met.

### Basic Requirements 
- NVIDIA GPU with CUDA 10.X.X capabilities
- Python 3.12.X+ 
- Anaconda 4.X+
- At least 25GB of storage available
- FFMPEG full release

### Setup
After making sure all of those requirements are alerady met you can start by cloning this repo

$ ```git clone https://github.com/scifi316/approximate-quilting-image-sequencer.git```

Download the correct version of necessary binaries from the [Releases](https://github.com/scifi316/approximate-quilting-image-sequencer/releases/tag/database) tab of the repo as well. 
- ___Current release data package is V1___

You will need to extract the binaries and install them in the correct locations.

You should be left with the files:
- frame_ids.npy
- frame_to_descriptor_indicies.npy
- individual_descriptor_faiss_index.bin
- input.zip

Create a folder in the root directory called `data`, then create a folder inside that called `images`

Extract the input.zip file and copy the contents into the `images` folder, while in the images directory. Create a folder called `quilted_output` and `source`, the current file structure should look like this:

- /root_folder/
    - src
      - database
      - ...
    - data
      - images
        - input
          - frame0000.png
          - frame0001.png
          - ...
        - quilted_output
        - source
    - tests
    - etc...

After creating and transferring the necessary files, navigate back to the root directory of the project where all the READMEs and LICENSE stuff is and copy the `frame_ids.npy`,
`frame_to_descriptor_indicies.npy`, and
`individual_descriptor_faiss_index.bin` to the folder.

_Now we should be able to start running some commands..._

In the root directory, open a terminal instance and run:

$ ```conda env create -f requirements.yml```

*Take a break as this may take a while to install the necessary python libraries, after it has finished installing, run: 

$ ```conda activate```

We should now have a proper working environemnt to run, test, and debug in. 

### Modifying and testing
__NOTE:__ Make sure you use the "activated" terminal to run the python files as it may fail to load the necessary libraries during certain steps.

Any changes made to the input database located in `/data/images/input` requires a rebuild of the database binaries and headings. To accomplish such, from the repository root run `python src/database/build_database.py`, this will take some time but should generate the new binaries/headings directly in the repository root.

By default this builds an `hnsw` Faiss index -- an approximate index that's
orders of magnitude faster to query than the old exact brute-force search,
with no GPU or training step required. Set the `FAISS_INDEX_TYPE`
environment variable to try a different index type (`flat`, `hnsw`,
`ivfflat`, `ivfpq`), e.g.:

```
FAISS_INDEX_TYPE=ivfflat python src/database/build_database.py
```

If you have a CUDA GPU set up per `requirements.yml`, `ivfflat` can also be
trained on the GPU with `FAISS_USE_GPU=1` -- this only speeds up the one-time
build step (~1.7s vs ~446s on this project's dataset); the resulting index is
converted back to a plain CPU index before being written, so no GPU is
required later to load or query it:

```
FAISS_INDEX_TYPE=ivfflat FAISS_USE_GPU=1 python src/database/build_database.py
```

See `benchmarks/RESULTS.md` for build time / size / query latency / recall
numbers for each type on this project's dataset, and how they were measured.

#### Descriptor type: sift (default) vs. tile

By default the database is built from sparse SIFT keypoints. Set
`QUILT_DESCRIPTOR_TYPE=tile` (plus matching `QUILT_CHUNK_WIDTH`/
`QUILT_CHUNK_HEIGHT`/`QUILT_THUMB_SIZE`) to build one dense descriptor per
grid tile instead:

```
QUILT_DESCRIPTOR_TYPE=tile QUILT_CHUNK_WIDTH=40 QUILT_CHUNK_HEIGHT=40 QUILT_THUMB_SIZE=4 \
FAISS_INDEX_TYPE=ivfflat FAISS_USE_GPU=1 python src/database/build_database.py
```

This exists for **fine-grained quilting** (smaller chunk sizes, more tiles
per output frame): SIFT keypoints don't scale down with chunk size, so
shrinking chunks with the default descriptor mostly leaves tiles with
nothing to match on and produces mostly-black output (see
`benchmarks/RESULTS.md` for the keypoint-density numbers). Tile descriptors
guarantee every chunk has exactly one to match on regardless of size.

**Tradeoff**: tile descriptors are a coarse per-tile color/luminance
signature, far less discriminative than SIFT's structural matching. In
practice this shows up as more *repetitive* tile content in the output --
visually distinct source frames with a similar overall color/brightness
collapse to similar matches, more so at small `QUILT_THUMB_SIZE`. Raising
`QUILT_THUMB_SIZE` (more spatial detail per tile, still tiny/fast relative
to SIFT) noticeably improves match variety at some further cost. There's no
single "right" setting here -- it's a real quality/speed tradeoff to tune
for your source material, not a strict upgrade over `sift` mode.

When building a `tile` database, the same `QUILT_CHUNK_WIDTH`/
`QUILT_CHUNK_HEIGHT`/`QUILT_THUMB_SIZE`/`QUILT_DESCRIPTOR_TYPE` must be set
when running `stitch.py` (see below) -- they define the descriptor's vector
space, and a mismatch between build and query silently produces nonsense
matches rather than an error. Chunk width/height must evenly divide both
the database frames' and target frames' resolution.

To also run the per-frame Faiss search on GPU (separate from the
GPU-accelerated database *build* above), set `QUILT_USE_GPU=1` when running
`stitch.py`; not supported when the database was built with `hnsw` (Faiss's
GPU backend doesn't implement it -- use `flat` or `ivfflat`).

To generate a video, make sure [FFMPEG](https://www.ffmpeg.org/download.html) is already installed on your system. This program works by sequentially generating each frame from a source video using the database of images given. 

This requires take your video source and splitting them into individual frames for processing. The simplest way is by opening a terminal in the `.../source` directory you should have already created. There copy your video to that directory, `"video_name.mp4"` and run the following FFMPEG command:

$ ```ffmpeg -i "video_name.mp4" -vf fps=[FPS of video] frame%04d.png```

Where `"video_name.mp4"` is the name of the video you want to use in MP4 format, and the `[FPS of video]` is the video's framerate in whole number format (24, 30, 60, etc.), this FPS is important for correctly generating the frames needed with FFMPEG. 

Depending on the specs of your PC and the video you use, this may take a moment for FFMPEG to parse and split the video into the correct amount of frames. 

__NOTE:__ Currently this program only supports frame generation with video resolutions of 1920x1080.

_Now_ we should be able to create some new frames, locate the file `stitch.py` in the directory `/src/tests/proto/`, if everything has been done correctly, running `stitch.py` through the CLI/Terminal should start generating new frames in the folder `quilted_output`. It will prompt you with the message `"Quilted all images"` when finished.

Frames are processed across multiple worker processes in parallel (each
target frame is independent), defaulting to `cpu_count() - 1` workers. Set
`QUILT_WORKERS=N` to override, e.g. `QUILT_WORKERS=1` to process frames one
at a time. Per-frame work is dominated by single-threaded SIFT feature
detection by default, so this scales close to linearly with available CPU
cores.

If the database was built with `QUILT_DESCRIPTOR_TYPE=tile` (see above), set
the *same* `QUILT_DESCRIPTOR_TYPE`/`QUILT_CHUNK_WIDTH`/`QUILT_CHUNK_HEIGHT`/
`QUILT_THUMB_SIZE` here too -- a mismatch silently produces nonsense matches
rather than an error, since both ends just see "a vector of the expected
dimension." Also set `QUILT_USE_GPU=1` to run the per-frame Faiss search on
GPU (requires a `flat`- or `ivfflat`-built database; not supported for
`hnsw`).

To rebuild the frames back into a video, navigate to the `"quilted_output"` directory and open a terminal instance, there we run:

$ ```ffmpeg -framerate [FPS of video] -i quilted_frame%04d.png -c:v libx264 -pix_fmt yuv420p out.mp4```

To include the original audio track, add it as a second input alongside
the frame sequence:

$ ```ffmpeg -framerate [FPS of video] -i quilted_frame%04d.png -i audio.mp3 -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest out.mp4```

Where `"FPS of video"` is the original framerate of the video you wanted to generate during the splitting process earlier; after a moment FFMPEG will have generated a file called `"out.mp4"`, this is your video, open it to make sure it was correctly generated by FFMPEG.

