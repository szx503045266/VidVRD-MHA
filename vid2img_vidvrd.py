import os
import glob
import subprocess
from tqdm import tqdm
from dataset import VidVRD

dataset = VidVRD('/home/szx/vidvrd-dataset', '/home/szx/vidvrd-dataset/videos', ['train', 'test'])
'''
print('Convert train videos to images...')
video_indices = dataset.get_index(split='train')
for vid in tqdm(video_indices):
    frame_dir = '/home/szx/vidvrd-dataset/frame-bbox/train/{}'.format(vid)
    os.makedirs(frame_dir)
    vpath = dataset.get_video_path(vid).replace('videos', 'videos-bbox/train')
    subprocess.run(['ffmpeg', '-i', vpath, '{}/%05d.JPEG'.format(frame_dir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frame_paths = glob.glob('{}/*'.format(frame_dir))
    frame_count = dataset.get_anno(vid)['frame_count']
    if frame_count != len(frame_paths):
        print('\tWarning: video {} cannot extract {} frames. Extracted {} frames'.format(vid, frame_count, len(frame_paths)))
'''     
print('Convert test videos to images...')
video_indices = dataset.get_index(split='test')
for vid in tqdm(video_indices):
    frame_dir = '/home/szx/vidvrd-dataset/frame-bbox/test/{}'.format(vid)
    os.makedirs(frame_dir)
    vpath = dataset.get_video_path(vid).replace('videos', 'videos-bbox/test')
    subprocess.run(['ffmpeg', '-i', vpath, '{}/%05d.JPEG'.format(frame_dir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frame_paths = glob.glob('{}/*'.format(frame_dir))
    frame_count = dataset.get_anno(vid)['frame_count']
    if frame_count != len(frame_paths):
        print('\tWarning: video {} cannot extract {} frames. Extracted {} frames'.format(vid, frame_count, len(frame_paths)))