'''
Copy this file under ByteTrack/tools/, * same folder to demo_track.py
'''

import argparse
import os
import os.path as osp
import time
import cv2
import torch
import pandas as pd
import numpy as np
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


# Simulate the args send to BYTETracker
class ARGS():
    def __init__(self):
        self.track_thresh = .5
        self.match_thresh = .8
        self.track_buffer = 30
        self.aspect_ratio_thresh = 1.6
        self.mot20 = False

args = ARGS()


class GT:
    def __init__(self,file_name):
        self.file_name = file_name
        self.read_gt()
        
    def read_gt(self):
        self.df = pd.read_csv(self.file_name,names=['frame_id','Trajectory_id','bbox_1','bbox_2','bbox_3','bbox_4','confidence','class','vis_ratio'])    
        print('In ground Truth')
        print('Trajectories:',self.df['Trajectory_id'].values.max())
        print('Bounding boxes:',self.df.index.max())
        
    def save_gt(self):
        save_file = self.file_name.split('.')[0]+'_processed.txt'
        self.df.to_csv(save_file,header=0,index=0)
        print('Processed ground truth saved to',save_file)
        return save_file

    def rm_frame(self,frame_id):
        indexes = np.array(self.df.index)[self.df['frame_id']==frame_id]
        self.df.drop(indexes,inplace=True)
        # print('Removed',len(indexes),'labels in frame',frame_id)


def get_sta_ids(video_path,sta_thres):

    camera = cv2.VideoCapture(video_path) 
    length = camera.get(cv2.CAP_PROP_FRAME_COUNT)

    (ret, lastFrame) = camera.read() # 1st

    lastFrame =  cv2.resize(cv2.cvtColor(lastFrame,cv2.COLOR_BGR2GRAY),[200,200])

    diff_list = []

    while camera.isOpened(): 

        (ret, frame) = camera.read() 

        if not ret: 
            break 

        frame = cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),[200,200])

        diff = frame - lastFrame # 2nd - 1st -> diff[0] -> diff[0] is the change of frame 2 -> id+2

        diff_abs = cv2.convertScaleAbs(diff)

        diff_list.append(diff_abs.mean())

        lastFrame = frame.copy() 

    diff_thres = np.sort(diff_list)[int(length * thres)]

    sta_frame_ids = []

    for id,diff in enumerate(diff_list):

        if diff < diff_thres:

            sta_frame_ids.append(id+2)

    return sta_frame_ids


def pred_video(exp_file,ckpt,video_path,gt_path,fuse=True,fp16=True,drop_each_frame=0,sta_thres=0):
    '''
    exp_file: expriment description file
    ckpt: the model, we might use pretrained/bytetrack_?_mot17.pth.tar
    '''
    if drop_each_frame!=0 and sta_thres!=0:
        logger.info("Only one of drop_each_frame or sta_thres could be  greater than 0".format(get_model_info(model, exp.test_size)))
    
    
    # get exp description file, model name set to be none
    exp = get_exp(exp_file, None)
    # make output and vis folder
    experiment_name = exp.exp_name
    output_dir = osp.join(exp.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    vis_folder = osp.join(output_dir, "track_vis")
    os.makedirs(vis_folder, exist_ok=True)

    device = torch.device("cuda")

    # load the model
    model = exp.get_model().to(device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    if fuse == True:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if fp16 == True:
        model = model.half()  # to FP16
        
    trt_file = None
    decoder = None

    predictor = Predictor(model, exp, trt_file, decoder,device,fp16) # predictor
    current_time = time.localtime() # read current time

    return imageflow_demo(predictor, vis_folder, current_time, video_path, gt_path, exp, drop_each_frame,sta_thres)


def imageflow_demo(predictor, vis_folder, current_time, video_path, gt_path, exp, drop_each_frame=0,sta_thres=0): 
    
    # read video info
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS) # read the FPS of video
    
    # read gt
    gt = GT(gt_path)
    
    # save path
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, video_path.split("/")[-1])
    logger.info(f"video save_path is {save_path}")

    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    # initialize the tracker and timer
    tracker = BYTETracker(args, frame_rate=30) # use the global variable args
    timer = Timer()
    frame_id = 0 + 1 # initialize the frame id +1 because gt's frame id start from 1
    results = []
    # time_start = time.time()

    # get static frame ids
    if sta_thres > 0:
        timer.tic() # the static frames calculation time should be contained into the process time
        sta_frame_ids = get_sta_ids(video_path,sta_thres)
        timer.toc() 
    
    while True:
        
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time))) # 
        
        
        # Process the video
        # remove one each drop_each_frame
        if drop_each_frame > 0 :        
            if frame_id % drop_each_frame  == 0:
                timer.tic() # simulate start the detecting
                ret_val, frame = cap.read() 
                gt.rm_frame(frame_id)
                frame_id += 1
                timer.toc() # simulate over the tracking
                continue
        
        # Process the video
        # remove all the static frame
        if sta_thres > 0:
            if   frame_id in sta_frame_ids:
                timer.tic() # simulate start the detecting
                ret_val, frame = cap.read() 
                gt.rm_frame(frame_id)
                frame_id += 1
                timer.toc() # simulate over the tracking
                continue
        
        # Process the video
            
        ret_val, frame = cap.read() 
        
        if ret_val: 
            '''
            if there are still frames in the video
            '''
            outputs, img_info = predictor.inference(frame, timer) # maybe modify the model here
            if outputs[0] is not None:
                '''
                if detected, then BYTETracker
                '''
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > 10 and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc() # returns time from last tic or from initialization
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                ) # plot the track on image
            else:
                '''
                nothing detected at all
                '''
                timer.toc() # total_time += diff, calls + 1 update the average 
                online_im = img_info['raw_img']
     
            vid_writer.write(online_im) # write into video
            ch = cv2.waitKey(1) # wait for the key board
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            '''
            if nothing in the video
            '''
            break
            
        frame_id += 1

    # calc average FPS
    avg_FPS = 1. / max(1e-5, timer.average_time)
    # time_end = time.time()
    # avg_FPS = fps/(time_end - time_start)

    # save output
    pred_file = osp.join(vis_folder, f"{timestamp}.txt")
    with open(pred_file, 'w') as f:
        f.writelines(results)
    logger.info(f"save results to {pred_file}")
    
    # save gt
    processed_gt = gt.save_gt()
    return pred_file,processed_gt,avg_FPS


class Predictor(object):
    '''
    The model detect the given image
    '''
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info