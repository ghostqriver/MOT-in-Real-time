# MOT-in-Real-time

This repo is the project for AML subject in ELTE.

The project's goal is to improve the [bytetrack](https://github.com/ifzhang/ByteTrack) 's prediction real time performance (FPS>30).

## How to use

1. Install bytetrack & dependencies
```shell
!pip3 install cython
!pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
!pip3 install cython_bbox
!pip3 install motmetrics

!git clone https://github.com/ifzhang/ByteTrack.git

%cd ByteTrack
!pip3 install -r requirements.txt
!python3 setup.py develop
```
2. Install our repo
```shell
%cd ..
!git clone https://github.com/ghostqriver/MOT-in-Real-time.git
%cd /MOT-in-Real-time/
!cp 'tracker.py' '/content/ByteTrack/tools'
```

3.(a) Use our defined video predictor
```shell
%cd /ByteTrack
import tools.tracker as tracker
exp_file = 'exps/example/mot/yolox_s_mix_det.py'
ckpt = 'pretrained/bytetrack_s_mot17.pth.tar'
video_path = # your video path
gt_path = # your gt path
tracker.pred_video(exp_file,ckpt,video_path,gt_path,fuse=True,fp16=True)
```
3.(b) Use our video processor to reduce training and inference time for bytetrack by reducing the framerate of the videos
from video_process import VideoPreprocessor

vp = VideoPreprocessor()
train_folder = 'path/to/train/set' # Works on set of videos
test_video = 'path/to/test/video'  # and individual videos as well
target_fps = 15 # reduce the original 30 fps videos to 15 fps
train_output_dict = vp.process_folder(train_folder, target_fps)  
test_output = vp.process_video(test_video, target_ps)
...
  run the bytetrack on the results
...
4. Calculate metrics

  After populating an accumulator via providing the ground truth file path and the output result file path, a big variety of multiple object tracking metrics (available to list via the ```list_available_metrics``` method) can be calculated through ```yield_metrics_from_accumulator``` in ```calc_metrics.py```


## Colab example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w_4PgAOQ-biOVtb2UCGuL2stxI_eCBpu?usp=sharing)
