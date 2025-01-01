import torch
import cv2
import numpy as np

# Load GFPGAN model
def load_gfpgan_model(model_path):
    from gfpgan import GFPGANer
    model = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2)
    return model

# Load CodeFormer model (optional)
def load_codeformer_model(model_path):
    try:
        from basicsr.archs.codeformer_arch import CodeFormer
        model = CodeFormer()
        model.load_state_dict(torch.load(model_path)['params_ema'])
        model.eval()
        return model
    except ImportError:
        print("CodeFormer not available, skipping.")
        return None

class SuperResolution:
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        if model_name == "GFPGAN":
            self.model = load_gfpgan_model(model_path)
        elif model_name == "CodeFormer":
            self.model = load_codeformer_model(model_path)
        else:
            raise ValueError("Invalid model name. Use 'GFPGAN' or 'CodeFormer'.")

    def apply_superres_to_bbox(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        cropped_frame = frame[y1:y2, x1:x2]
        if self.model_name == "GFPGAN":
            enhanced_crop = self.apply_gfpgan(cropped_frame)
        elif self.model_name == "CodeFormer" and self.model:
            enhanced_crop = self.apply_codeformer(cropped_frame)
        else:
            enhanced_crop = cropped_frame  # No enhancement
        frame[y1:y2, x1:x2] = enhanced_crop
        return frame

    def apply_gfpgan(self, image):
        _, _, output = self.model.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
        return output

    def apply_codeformer(self, image):
        with torch.no_grad():
            image_torch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().div(255).to(next(self.model.parameters()).device)
            output = self.model(image_torch, w=0.5)
            output_np = (output.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return output_np
