
import av
import io
from transformers import Sam2VideoModel, Sam2VideoProcessor
import torch
import PIL
import colorsys
import numpy as np
import math
import time
import copy
from fastapi import UploadFile
from typing import BinaryIO




device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-tiny", cache_dir='./models').to(device, dtype=torch.bfloat16)
processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-tiny", cache_dir='./models')


def make_palette(n:int):
    # outputs a rainbow of n colours
    hsv = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
    rgb = [colorsys.hsv_to_rgb(*x) for x in hsv]
    # flatten list
    rgb = [x for xs in rgb for x in xs]
    # fit to 8 bit int range and make the first colour black
    rgb = [0,0,0] + [math.floor(255*x) for x in rgb]
    return rgb

def median_index(a):
    # finds the median index of an array time * objects * summed_dimension
    midpoint = a.sum(axis = 2) / 2
    cumulative = a.cumsum(axis = 2)
    inflection = cumulative > midpoint[:, :, None] # reshape midpoint to 3D
    index = inflection.argmax(axis = 2)
    return index



def centre_of_mass(a):
    # calculates the central point of an array of masks
    x = a.sum(axis=2) # shape time * objects * x
    y = a.sum(axis=3) # shape time * objects * y
    return median_index(x), median_index(y)



def move_ownership(old_file: UploadFile) -> UploadFile:
    # ugly workaround to keep the UploadFile around
    new_file = UploadFile(
        file=old_file.file,
        size=old_file.size,
        filename=old_file.filename,
        headers=old_file.headers,
    )
    old_file.file = BinaryIO()
    return new_file

class CropSession:

    video_file = None
    frame_iter = None
    time_downscale_factor = 5
    container = None
    first_frame = None
    video_frames = None
    inference_session = None
    frame_index = 0
    coords = None

    def reset(self):
        self.video_file = None
        self.frame_iter = None
        self.time_downscale_factor = 1
        self.container = None
        self.first_frame = None
        self.video_frames = None
        self.inference_session = None
        self.frame_index = 0
        self.coords = None
    def __init__(self):
        pass

    def load_frames(self, frame_count):
        self.video_frames = []
        for i in range(frame_count * self.time_downscale_factor):
            frame = self.frame_iter.__next__()
            if i % self.time_downscale_factor == 0:
                self.video_frames.append(frame.to_image())

    def save_frame(self, index: int):
        img_byte_arr = io.BytesIO()
        self.video_frames[index].save(img_byte_arr, format='jpeg')
        img_byte_arr.seek(0)
        self.first_frame = img_byte_arr

    def load_video_file(self, video_file):
        self.reset()
        self.frame_index = 0
        self.video_file = move_ownership(video_file)
        self.container = av.open(self.video_file.file)
        self.frame_iter = self.container.decode(video=0)
        self.masks = []
        # load first x frames
        self.load_frames(1)
        # save first frame to send to client
        self.save_frame(0)
        #self.video_frames = np.array(self.video_frames)
        # should yield here and move to new thread to improve performance
        # start inference session
        self.inference_session = processor.init_video_session(
            video=np.array(self.video_frames),
            inference_device=device,
            dtype=torch.bfloat16,
        )
    
    def generate_masks(self, points):
        # to do: remove previous objects before running
        ann_frame_idx = 0
        ann_obj_id = [x for x in range(len(points))]
        labels = [[[1] for x in points]]
        points = [[[x] for x in points]]
        processor.add_inputs_to_inference_session(
            inference_session=self.inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_points=points,
            input_labels=labels,
        )
        # Segment the object on the first frame
        outputs = model(
            inference_session=self.inference_session,
            frame_idx=ann_frame_idx,
        )
        masks = processor.post_process_masks(
            [outputs.pred_masks], original_sizes=[[self.inference_session.video_height, self.inference_session.video_width]], binarize=True
        )[0]
        masks = masks.cpu()
        final_mask = np.zeros_like(masks[0,0], dtype=np.int32)
        for i in range(masks.shape[0]):
            layer = masks[i][0].numpy()
            final_mask += layer * (i+1)
        img = PIL.Image.fromarray(final_mask.astype(np.uint8), mode='P')
        palette = make_palette(masks.shape[0])
        img.putpalette(palette)
        img = img.convert('RGB')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='jpeg')
        img_byte_arr.seek(0)
        self.masks = [img_byte_arr]
        
    def track(self):
        self.save_frame(-1)
        masks = []
        for sam2_video_output in model.propagate_in_video_iterator(self.inference_session, start_frame_idx=self.frame_index):
            print("starting frame: ", sam2_video_output.frame_idx)
            video_res_masks = processor.post_process_masks(
                [sam2_video_output.pred_masks], original_sizes=[[self.inference_session.video_height, self.inference_session.video_width]], binarize=True
            )[0]
            masks.append(video_res_masks[:,0].cpu()) # for some reason there's an extra dimension to get rid of
            print("ending frame: ", sam2_video_output.frame_idx, time.time())
            self.frame_index += 1
        x, y = centre_of_mass(np.array(masks))
        # rezip the two arrays to time * objects * xy coords
        coords = np.stack((x,y), axis=2)
        # move objects first to make easier to work with in JS
        coords = coords.swapaxes(0, 1)
        if self.coords is None:
            self.coords = coords
        else:
            self.coords = np.concatenate([self.coords,coords], axis = 1) # add new coordinates on the time axis
        self.load_frames(2)
        for frame in np.array(self.video_frames):
            image = processor(images=frame, device=device, return_tensors='pt')
            self.inference_session.add_new_frame(image.pixel_values[0])
        
    def crop(self, n_frames: int, size: int):
        self.container.seek(0)
        self.frame_iter = self.container.decode(video=0)
        self.video_frames = []
        self.load_frames(n_frames)
        object_coords = self.coords[0] # just handle the first click so far
        frames = np.array(self.video_frames)
        cropped_frames = []
        for i, frame in enumerate(frames):
            x = object_coords[i][0]
            y = object_coords[i][1]
            print(x,y)
            print(frame.shape)
            cropped_frames.append(frame[y-size:y+size, x-size:x+size])
            print(cropped_frames[-1].shape)
        cropped_frames = np.array(cropped_frames,dtype=np.uint8)
        print(cropped_frames.shape)
        write_container = av.open('test.mp4', mode='w')
        stream = write_container.add_stream('mpeg4', rate=24, options={'b:v': '192000', 'maxrate': '192000'})
        stream.width = size*2
        stream.height = size*2
        stream.pix_fmt = 'yuv420p'
        for img in cropped_frames:
            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            for packet in stream.encode(frame):
                write_container.mux(packet)
        # Flush stream
        for packet in stream.encode():
            write_container.mux(packet)

        # Close the file
        write_container.close()

