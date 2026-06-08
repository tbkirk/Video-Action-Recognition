
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

def interpolate_coords(coords, downsample, total_frames):
    # linearly interpolate coordinates to match target length
    x = coords[:, 0]
    y = coords[:, 1]
    index = np.arange(0, len(x)*downsample, downsample)
    resampled_index = np.arange(0, total_frames)
    x_interp = np.interp(resampled_index, index, x)
    y_interp = np.interp(resampled_index, index, y)
    return np.stack((x_interp, y_interp), axis=1)

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

def get_offset(position: int, size: int, limit: int) -> int:
    offset = 0
    if position - size < 0:
        offset = 0 - (position - size)
    elif position + size >= limit:
        offset = limit - (position + size) # might be off by one?
    return offset

def get_boundaries(x: int, y: int, size: int, x_limit: int, y_limit: int):
    x = x + get_offset(x, size, x_limit)
    y = y + get_offset(y, size, y_limit)
    return [x-size, x+size, y-size, y+size]


class CropSession:

    video_file = None
    frame_iter = None
    time_downscale_factor = None
    container = None
    first_frame = None
    video_frames = None
    inference_session = None
    frame_index = 0
    coords = None
    total_frames = None

    def reset(self):
        self.video_file = None
        self.frame_iter = None
        self.container = None
        self.first_frame = None
        self.video_frames = None
        self.inference_session = None
        self.frame_index = 0
        self.coords = None
    def __init__(self):
        self.time_downscale_factor = 10
        pass

    def load_frames(self, frame_count, downsample=None):
        if downsample is None:
            downsample = self.time_downscale_factor
        self.video_frames = []
        for i in range(frame_count * downsample):
            frame = self.frame_iter.__next__()
            if i % downsample == 0:
                self.video_frames.append(frame.to_image())
                print("loaded frame ", i)

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
        self.total_frames = self.container.streams.video[0].frames
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
            #max_vision_features_cache_size=10,
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
        # todo use server send events to send masks as they are generated rather than waiting until the end
        self.save_frame(-1)
        print('loading frames for tracking')
        print("total frames in video: ", self.total_frames)
        print("total frames to process: ", (self.total_frames // self.time_downscale_factor))
        print(self.time_downscale_factor)
        self.load_frames((self.total_frames // self.time_downscale_factor) - 1) # might be off by one?
        print('frames loaded, starting tracking')
        for i, frame in enumerate(np.array(self.video_frames)):
            if i % 10 == 0:
                print("processing frame ", i)
            image = processor(images=frame, device=device, return_tensors='pt')
            self.inference_session.add_new_frame(image.pixel_values[0])
        print('finished processing frames')
        masks = []
        # todo break into shorter videos to improve performance and reduce memory usage
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

        
    def crop(self, n_frames: int, size: int):
        self.container.seek(0)
        n_frames = self.total_frames # todo handle different lengths
        print("total frames: ", n_frames)
        self.frame_iter = self.container.decode(video=0)
        self.video_frames = []
        self.load_frames(self.total_frames, downsample=1) # load all frames
        object_coords = []
        for coords in self.coords:
            object_coords.append(interpolate_coords(coords, self.time_downscale_factor, self.total_frames))
        object_coords = np.array(object_coords)
        print("coords shape: ", object_coords.shape)
        object_coords = object_coords.astype(int)
        frames = np.array(self.video_frames)
        cropped_frames = [[] for x in range(object_coords.shape[0])] # list of lists to hold cropped frames for each object
        for i, frame in enumerate(frames):
            for j, coord in enumerate(object_coords):
                x = coord[i][0]
                y = coord[i][1]
                print(x,y)
                print(frame.shape)
                boundaries = get_boundaries(x, y, size, frame.shape[1], frame.shape[0])
                cropped_frames[j].append(frame[boundaries[2]:boundaries[3], boundaries[0]:boundaries[1]])
                print(cropped_frames[j][-1].shape)
        cropped_frames = np.array(cropped_frames,dtype=np.uint8)
        print(cropped_frames.shape)
        for i in range(cropped_frames.shape[0]):
            write_container = av.open('C:\\Users\\hmz574\\Downloads\\'+str(i)+'.mp4', mode='w')
            stream = write_container.add_stream('mpeg4', rate=24)
            stream.width = size*2
            stream.height = size*2
            stream.pix_fmt = 'yuv420p'
            stream.bit_rate = 1000000 # TODO work out how to vary this by required bitrate
            for img in cropped_frames[i]:
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                for packet in stream.encode(frame):
                    write_container.mux(packet)
            # Flush stream
            for packet in stream.encode():
                write_container.mux(packet)

            # Close the file
            write_container.close()

