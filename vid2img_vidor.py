import os
import glob
import subprocess
from tqdm import tqdm
from dataset import VidOR

dataset = VidOR('/home/szx/vidor-dataset/annotation', '/home/szx/vidor-dataset/video', ['training', 'validation'], low_memory=True)
'''
print('Convert training videos to images...')
video_indices = dataset.get_index(split='training')
for vid in tqdm(video_indices):
    frame_dir = '/home/szx/vidor-dataset/frame/training/{}'.format(vid)
    os.makedirs(frame_dir)
    vpath = dataset.get_video_path(vid).replace('video/training-video', 'video-bbox/training')  #TODO
    subprocess.run(['ffmpeg', '-i', vpath, '{}/%05d.JPEG'.format(frame_dir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frame_paths = glob.glob('{}/*'.format(frame_dir))
    frame_count = dataset.get_anno(vid)['frame_count']
    if frame_count != len(frame_paths):
        print('\tWarning: video {} cannot extract {} frames. Extracted {} frames'.format(vid, frame_count, len(frame_paths)))
'''     
print('Convert validation videos to images...')
video_indices = dataset.get_index(split='validation')
for vid in tqdm(video_indices):
    frame_dir = '/home/szx/vidor-dataset/frame/validation/{}'.format(vid)
    os.makedirs(frame_dir)
    vpath = dataset.get_video_path(vid).replace('video/validation-video/*/', 'video-bbox/validation') #TODO
    subprocess.run(['ffmpeg', '-i', vpath, '{}/%05d.JPEG'.format(frame_dir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frame_paths = glob.glob('{}/*'.format(frame_dir))
    frame_count = dataset.get_anno(vid)['frame_count']
    if frame_count != len(frame_paths):
        print('\tWarning: video {} cannot extract {} frames. Extracted {} frames'.format(vid, frame_count, len(frame_paths)))