import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy
from musetalk.utils.utils import get_file_type, get_video_fps, datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model
import shutil

# Conditional imports for super-resolution modules to avoid conflicts
def load_gfpgan():
    from gfpgan import GFPGANer
    model_path = 'F:\\VideoDubber\\MuseTalk\\enhance_models\\GFPGANv1.3.pth'
    print(f"Loading GFPGAN model from {model_path}")
    model = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2)
    return model

def enhance_with_gfpgan(model, image):
    _, _, output = model.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
    return output

def load_codeformer():
    from basicsr.archs.codeformer_arch import CodeFormer
    model_path = 'F:\\VideoDubber\\MuseTalk\\enhance_models\\codeformer.pth'
    print(f"Loading CodeFormer model from {model_path}")
    model = CodeFormer()
    model.load_state_dict(torch.load(model_path)['params_ema'])
    model.eval()
    return model

def enhance_with_codeformer(model, image):
    with torch.no_grad():
        image_torch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().div(255).to(next(model.parameters()).device)
        output = model(image_torch, w=0.5)  # Adjust w for fidelity
        output_np = (output.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return output_np

# Load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)

# Load super-resolution model based on input choice
def load_superres_model(model_name="GFPGAN"):
    if model_name == "GFPGAN":
        return load_gfpgan()
    elif model_name == "CodeFormer":
        return load_codeformer()
    else:
        raise ValueError("Invalid super-resolution model name")

# Enhance image based on the chosen super-resolution model
def enhance_image(model, model_name, image):
    if model_name == "GFPGAN":
        return enhance_with_gfpgan(model, image)
    elif model_name == "CodeFormer":
        return enhance_with_codeformer(model, image)
    else:
        raise ValueError("Invalid super-resolution model name")

# Calculate resolution ratio
def resolution_ratio(original_bbox, generated_frame):
    # Extract bounding box coordinates (x1, y1, x2, y2)
    x1, y1, x2, y2 = original_bbox
    
    # Calculate original width and height from bounding box
    original_width = x2 - x1
    original_height = y2 - y1
    
    # Get dimensions of the generated (super-resolved) frame
    output_height, output_width = generated_frame.shape[:2]
    
    # Calculate resolution ratio (width and height)
    width_ratio = output_width / original_width
    height_ratio = output_height / original_height
    
    return width_ratio, height_ratio


def apply_superres(model, model_name, frame, bbox, ratio):
    x1, y1, x2, y2 = bbox
    # Crop the generated frame
    generated_crop = frame[y1:y2, x1:x2]
    
    # Apply super-resolution if the resolution is poorer
    if ratio[0] < 1 or ratio[1] < 1:  # If resolution is poorer
        generated_crop = enhance_image(model, model_name, generated_crop)
    
    # Resize the super-resolved image to match the bounding box size
    height, width = y2 - y1, x2 - x1
    generated_crop_resized = cv2.resize(generated_crop, (width, height), interpolation=cv2.INTER_LANCZOS4)
    
    # Place the super-resolved image back into the frame
    frame[y1:y2, x1:x2] = generated_crop_resized
    return frame

@torch.no_grad()
def main(args):
    global pe
    if args.use_float16 is True:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
    
    # Load the super-resolution model
    superres_model = load_superres_model(args.superres)  # 'GFPGAN' or 'CodeFormer'
    
    video_path = args.video_input
    audio_path = args.audio_input
    bbox_shift = args.bbox_shift

    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename  = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    result_img_save_path = os.path.join(args.result_dir, output_basename)  # related to video & audio inputs
    crop_coord_save_path = os.path.join(result_img_save_path, input_basename + ".pkl")  # only related to video input
    os.makedirs(result_img_save_path, exist_ok=True)

    if args.output_vid_name is None:
        output_vid_name = os.path.join(args.result_dir, output_basename + ".mp4")
    else:
        output_vid_name = os.path.join(args.result_dir, args.output_vid_name)
    
    # Extract frames from source video
    if get_file_type(video_path) == "video":
        save_dir_full = os.path.join(args.result_dir, input_basename)
        os.makedirs(save_dir_full, exist_ok=True)
        cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
        os.system(cmd)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    elif get_file_type(video_path) == "image":
        input_img_list = [video_path, ]
        fps = args.fps
    elif os.path.isdir(video_path):  # input img folder
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args.fps
    else:
        raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")

    # Extract audio feature
    whisper_feature = audio_processor.audio2feat(audio_path)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)

    # Preprocess input images
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        print("using extracted coordinates")
        with open(crop_coord_save_path, 'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        print("extracting landmarks...time consuming")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
    
    # Prepare input latent list
    i = 0
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)
    
    # Smoothing and inference batch processing
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    
    print("start inference")
    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size)
    res_frame_list = []
    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / batch_size)))): 
        audio_feature_batch = torch.from_numpy(whisper_batch)
        audio_feature_batch = audio_feature_batch.to(device=unet.device, dtype=unet.model.dtype)
        audio_feature_batch = pe(audio_feature_batch)
        latent_batch = latent_batch.to(dtype=unet.model.dtype)
        
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
    
    print("pad talking image to original video")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i % len(coord_list_cycle)]
        ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
        x1, y1, x2, y2 = bbox
        
        # Calculate resolution ratio
        ratio = resolution_ratio(bbox, res_frame)
        
        # Apply super-resolution if necessary
        res_frame = apply_superres(superres_model, args.superres, res_frame, bbox, ratio)
        
        ori_frame[y1:y2, x1:x2] = res_frame
        
        output_frame_path = os.path.join(result_img_save_path, f'{i:08d}.png')
        cv2.imwrite(output_frame_path, ori_frame[:, :, ::-1])
    
    # Combine frames into video
    print("combine frames to video")
    cmd = f'ffmpeg -y -framerate {fps} -i {result_img_save_path}/%08d.png -i {audio_path} -shortest {output_vid_name}'
    os.system(cmd)
    
    # Cleanup
    if not args.save_frame:
        shutil.rmtree(result_img_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Musetalk Video Super-Resolution Enhancement')
    parser.add_argument('-sr', '--superres', type=str, required=True, choices=['GFPGAN', 'CodeFormer'], help='Super-resolution model to use')
    parser.add_argument('-iv', '--video_input', type=str, required=True, help='Input video file')
    parser.add_argument('-ia', '--audio_input', type=str, required=True, help='Input audio file')
    parser.add_argument('-o', '--output_vid_name', type=str, help='Output video file name')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--fps', type=int, default=25, help='Frames per second for the output video')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--use_float16', action='store_true', help='Use float16 precision for inference')
    parser.add_argument('--save_frame', action='store_true', help='Save intermediate frames')
    parser.add_argument('--use_saved_coord', action='store_true', help='Use saved coordinates for face landmarks')
    parser.add_argument('--bbox_shift', type=int, default=10, help='Shift for bounding box')
    
    args = parser.parse_args()
    main(args)
