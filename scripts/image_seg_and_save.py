import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import hydra
from omegaconf import OmegaConf
import os

# Use bfloat16 for the entire script
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # Turn on tfloat32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    ax.imshow(img)

# Load the image from a hardcoded path
image_path = '/home/arghya/segment-anything-2/data/left_000000.jpg'
image = Image.open(image_path)
image = np.array(image.convert("RGB"))

# Initialize the SAM2 model
# sam2_checkpoint = "/home/arghya/segment-anything-2/checkpoints/sam2_hiera_base_plus.pt"
sam2_checkpoint = "/home/arghya/segment-anything-2/checkpoints/sam2_hiera_large.pt"
# sam2_checkpoint = "/home/arghya/segment-anything-2/checkpoints/sam2_hiera_small.pt"
# sam2_checkpoint = "/home/arghya/segment-anything-2/checkpoints/sam2_hiera_tiny.pt"

# model_cfg = "sam2_hiera_b+.yaml"
model_cfg = "sam2_hiera_l.yaml"
# model_cfg = "sam2_hiera_s.yaml"
# model_cfg = "sam2_hiera_t.yaml"

# Add configuration path to Hydra
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path="../sam2_configs", version_base=None)

sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)

# Measure inference time
start_time = time.time()
# Generate masks
masks = mask_generator.generate(image)
end_time = time.time()

inference_time = end_time - start_time
print(f"Inference time: {inference_time:.4f} seconds")

print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')

# Save the resulting image
output_path = os.path.join('/home/arghya/segment-anything-2/data', 'output_left_000000.png')
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.show()

print(f"Saved the output image to {output_path}")
