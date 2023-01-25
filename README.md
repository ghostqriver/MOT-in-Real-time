# MOT-in-Real-time

This repo is the project for AML subject in ELTE.

The project's goal is to improve the [bytetrack](https://github.com/ifzhang/ByteTrack) 's prediction real time performance (FPS>30).

Get access to our team [report](https://docs.google.com/document/d/1CL5NLqxpi42jAJE1RuJuDSf8qFIHfQXw/edit?usp=sharing&ouid=109729230889422611512&rtpof=true&sd=true).

## How to use

1.(a) Install bytetrack & dependencies
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
1.(b) Install bot-sort & dependencies
```shell
%cd /content
!git clone https://github.com/NirAharon/BoT-SORT.git
%cd BoT-SORT
!pip3 install -r requirements.txt
!pip3 install faiss-cpu
!pip3 install faiss-gpu
!python3 setup.py develop
```
2. Install our repo
```shell
%cd ..
!git clone https://github.com/ghostqriver/MOT-in-Real-time.git
%cd /MOT-in-Real-time/
!cp 'tracker.py' '/content/ByteTrack/tools'
!cp 'byte_tracker_yolov7.py' '/content/ByteTrack/yolox/tracker'
!cp 'demo_track_yolov7.py' '/content/ByteTrack/tools'
!cp 'tracker_BS.py' '/content/BoT-SORT/tools'
```

3.(a) Use our defined video predictor + frame reduce function
There are two important parameters 

```drop_each_frame```: integer, when > 0, the frame would be reduced each ```drop_each_frame``` frames.

```sta_thres```: float (0,1), when > 0, intotal ```sta_thres``` * total frames number of most static frames would be reduced.

note that ```drop_each_frame``` and ```sta_thres``` could not > 0 at the same time.
```shell
%cd /ByteTrack
import tools.tracker as tracker

exp_file = 'exps/example/mot/yolox_s_mix_det.py'
ckpt = 'pretrained/bytetrack_s_mot17.pth.tar'
video_path = # your video path
gt_path = # your gt path
pred_file,processed_gt,avg_FPS = tracker.pred_video(exp_file,ckpt,video_path,gt_path,fuse=True,fp16=True,drop_each_frame=0,sta_thres=0.5)
```
3.(b) Use our video processor to reduce training and inference time for bytetrack by reducing the framerate of the videos
from video_process import VideoPreprocessor
```shell
vp = VideoPreprocessor()
train_folder = 'path/to/train/set' # Works on set of videos
test_video = 'path/to/test/video'  # and individual videos as well
target_fps = 15 # reduce the original 30 fps videos to 15 fps
train_output_dict = vp.process_folder(train_folder, target_fps)  
test_output = vp.process_video(test_video, target_fps)
...
  run the bytetrack on the results
...
```
4. Calculate and plot metrics

  To calculate some multiple object tracking metrics and to plot the calculation results, use the ```calculate_metrics_for_output``` and the ```plot_results``` method accordingly, from ```calc_metrics.py``` . The metrics have been aligned with what is reported by [MOTChallenge](https://motchallenge.net/) benchmarks. In the metric calculation procedure the ```motmetrics``` open source, third party library was used. Link to repo: https://github.com/cheind/py-motmetrics . For plotting,  the popular```matplotlib``` library was used. Link to repo: https://github.com/matplotlib/matplotlib .

The ```calculate_metrics_for_output``` method requires two input parameters, one for the ground truth file and one for the model's calculated output file. Optionally, one can provide a list of the desired metrics or set true for the ```all_metrics``` boolean parameter to yield all the available metrics. 

The ```plot_results``` method requires three parameter inputs: the calculated metrics for two models (e.g. the original one and an FPS enhanced one) and a list of the to-be-plotted metrics.




## Colab example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w_4PgAOQ-biOVtb2UCGuL2stxI_eCBpu?usp=sharing)
