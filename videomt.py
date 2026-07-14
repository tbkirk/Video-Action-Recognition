#import matplotlib.pyplot as plt
import numpy as np
import torch

from transformers import AutoModelForUniversalSegmentation, AutoVideoProcessor
from transformers.video_utils import load_video

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model_id = "tue-mps/videomt-dinov2-small-ytvis2019"
processor = AutoVideoProcessor.from_pretrained(model_id)
model = AutoModelForUniversalSegmentation.from_pretrained(model_id,).to(device, dtype=torch.bfloat16)

video_url = "https://huggingface.co/spaces/LanguageBind/Video-LLaVA/resolve/c9c92acb2515b23ee07dbed3d07dd7dea174f56f/examples/sample_demo_18.mp4"
# Sample 8 frames to keep the example lightweight.
video_frames, _ = load_video(video_url, num_frames=120)
video_frames = video_frames[0:8]

inputs = processor(videos=[video_frames], return_tensors="pt").to(model.device)

with torch.inference_mode():
    outputs = model(**inputs)

original_height, original_width = video_frames[0].shape[:2]
target_sizes = [(original_height, original_width)] * len(video_frames)

results = processor.post_process_instance_segmentation(
    outputs,
    target_sizes=target_sizes,
)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for idx, (ax, frame, result) in enumerate(zip(axes.flatten(), video_frames, results)):
    ax.imshow(frame)
    seg = result["segmentation"].cpu().numpy()
    masked = np.ma.masked_where(seg == -1, seg)
    ax.imshow(masked, alpha=0.6, cmap="tab20")
    ax.set_title(f"Frame {idx}")
    ax.axis("off")
plt.suptitle("Video Instance Segmentation")
plt.tight_layout()
plt.show()