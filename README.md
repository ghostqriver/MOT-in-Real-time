# MOT-in-Real-time

This repo is the project for AML subject in ELTE.

The project's goal is to improve the [bytetrack](https://github.com/ifzhang/ByteTrack) 's prediction real time performance (FPS>30).

## How to use

1. Install bytetrack & dependencies
```shell
!pip3 install cython
!pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
!pip3 install cython_bbox

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

3. Use our defined video predictor
```shell
%cd /ByteTrack
import tools.tracker as tracker
exp_file = 'exps/example/mot/yolox_s_mix_det.py'
ckpt = 'pretrained/bytetrack_s_mot17.pth.tar'
video_path = # your video path
gt_path = # your gt path
tracker.pred_video(exp_file,ckpt,video_path,gt_path,fuse=True,fp16=True)
```
