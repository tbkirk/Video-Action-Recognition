import av
import io
from transformers import AutoModelForUniversalSegmentation, Sam2VideoModel, Sam2VideoProcessor
import torch
import PIL
import colorsys
import numpy as np
import math
import time
import copy
from fastapi import UploadFile
from typing import BinaryIO


#VidEoMT 
from transformers import AutoModelForUniversalSegmentation, AutoVideoProcessor
from transformers.video_utils import load_video

# load model/init

# generate masks
# masks  = tensor[batch_size, num_channels, height, width]
# return masks

# track

device = "cuda" if torch.cuda.is_available() else "cpu"

def split_masks(masks, num_objects):
    split_masks = []
    for i in range(num_objects):
        split_masks.append(masks == i)
    return split_masks

class VidEoMTModel:
    def __init__(self, model_id="tue-mps/videomt-dinov2-small-ytvis2019"):
        self.model_id = model_id
        self.processor = AutoVideoProcessor.from_pretrained(model_id)
        self.model = AutoModelForUniversalSegmentation.from_pretrained(model_id,).to(device, dtype=torch.bfloat16)
    def generate_masks(self, video_frames):
        print("video frames shape: ", np.array(video_frames).shape) # (frames, height, width, channels)
        inputs = self.processor(videos=[video_frames], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        original_height, original_width = video_frames[0].shape[:2]
        target_sizes = [(original_height, original_width)] * len(video_frames)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            target_sizes=target_sizes,
        ) # array of dicts of length frames with keys: "segmentation", "segments_info"
        number_of_objects = len(results[0]["segments_info"])
        print("objects detected:", number_of_objects)
        masks = [x['segmentation'].to(dtype=torch.int8, device='cpu') for x in results] # shape: frames, height, width
        # old code to match the sam model code, but we can just return the masks as is since they are already in the correct shape
        #masks = []
        #for frame in results:
        #    masks.append(split_masks(frame["segmentation"], number_of_objects))
        #    # shape: frames, masks, height, width
        #masks = torch.tensor(masks, dtype=torch.int32)
        #masks = torch.permute(masks, (1, 0, 2, 3)) # shape: masks, frames, height, width
        # return shape [masks, frames, height, width]
        masks = np.array(masks)
        return masks


model = VidEoMTModel()
video_url = "https://huggingface.co/spaces/LanguageBind/Video-LLaVA/resolve/c9c92acb2515b23ee07dbed3d07dd7dea174f56f/examples/sample_demo_18.mp4"
# Sample 8 frames to keep the example lightweight.
video_frames, _ = load_video(video_url, num_frames=120)
video_frames = video_frames[0:8]

a = model.generate_masks(video_frames)
