# Overview
This repository contains two Python scripts designed for super-resolution enhancement of video frames using the GFPGAN and CodeFormer models. The scripts provide functionality to apply super-resolution to bounding boxes within video frames, ensuring improved visual quality for facial regions.

## Script 1: `superres.py`
This script contains the implementation for loading and applying the GFPGAN and CodeFormer models to enhance images.

### Dependencies
- `"torch"`
- `"cv2"`
- `"numpy"`
- `"gfpgan"` (for GFPGAN model)
- `"basicsr"` (for CodeFormer model)

### Functions

#### `load_gfpgan_model(model_path)`
Loads the GFPGAN model from the specified path.

#### `load_codeformer_model(model_path)`
Loads the CodeFormer model from the specified path, if available.

### **SuperResolution class**

#### `__init__(self, model_name, model_path)`
Initializes the super-resolution model.

#### `apply_superres_to_bbox(self, frame, bbox)`
Applies super-resolution to the specified bounding box within a frame.

#### `apply_gfpgan(self, image)`
Applies the GFPGAN model to an image.

#### `apply_codeformer(self, image)`
Applies the CodeFormer model to an image.

---

## Script 2: `main_script.py`
This script integrates the super-resolution functionality into a pipeline for processing video and audio files, extracting frames, applying enhancements, and reconstructing the video.

### Dependencies
- `"argparse"`
- `"os"`
- `"numpy"`
- `"cv2"`
- `"torch"`
- `"glob"`
- `"pickle"`
- `"tqdm"`
- `"copy"`
- `"musetalk"` (custom utility functions for video processing)

### Functions

#### `load_gfpgan()`
Loads the GFPGAN model from a predefined path.

#### `enhance_with_gfpgan(model, image)`
Enhances an image using the GFPGAN model.

#### `load_codeformer()`
Loads the CodeFormer model from a predefined path.

#### `enhance_with_codeformer(model, image)`
Enhances an image using the CodeFormer model.

#### `load_superres_model(model_name)`
Loads the specified super-resolution model (`"GFPGAN"` or `"CodeFormer"`).

#### `enhance_image(model, model_name, image)`
Enhances an image based on the specified model.

#### `resolution_ratio(original_bbox, generated_frame)`
Calculates the resolution ratio between the original bounding box and the generated frame.

#### `apply_superres(model, model_name, frame, bbox, ratio)`
Applies super-resolution to a frame if the resolution is poorer.

#### `main(args)`
Main function that handles the video and audio processing, enhancement, and reconstruction.

---

## Usage

To run the main script, use the following command:

```bash
python main_script.py --superres <model_name> --video_input <video_path> --audio_input <audio_path> [options]

## Arguments

The script accepts several arguments for configuring the processing:

- `--superres`: Specifies the super-resolution model to use. Options are:
  - `GFPGAN`
  - `CodeFormer`
  
- `--video_input`: Path to the input video file.

- `--audio_input`: Path to the input audio file.

- `--output_vid_name`: (Optional) Name for the output video file.

- `--result_dir`: Directory where results will be saved (default is `results`).

- `--fps`: Frames per second for the output video (default is `25`).

- `--batch_size`: The batch size for inference (default is `8`).

- `--use_float16`: Flag to use float16 precision for inference.

- `--save_frame`: Flag to save intermediate frames during processing.

- `--use_saved_coord`: Flag to use previously saved coordinates for face landmarks.

- `--bbox_shift`: Adjusts the bounding box shift (default is `10`).

## Example Command

To process a video and audio, apply super-resolution using the `GFPGAN` model, and save the output to a specified directory, use the following command:

```bash
python main_script.py --superres GFPGAN --video_input input_video.mp4 --audio_input input_audio.wav --result_dir output --fps 30 --batch_size 4 --use_float16 --save_frame --bbox_shift 15
