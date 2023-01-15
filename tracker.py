'''
Copy this file under ByteTrack/tools/, * same folder to demo_track.py
'''

import argparse
import os
import os.path as osp
import time
import cv2
import torch

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


def pred_video(exp_file,ckpt,video_path,gt_path,fuse=True,fp16=True):
    '''
    exp_file: expriment description file
    ckpt: the model, we might use pretrained/bytetrack_?_mot17.pth.tar
    '''
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

    imageflow_demo(predictor, vis_folder, current_time, video_path, gt_path, exp)


def imageflow_demo(predictor, vis_folder, current_time, video_path, gt_path, exp): 
    
    # read video info
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS) # read the FPS of video
    
    # save path
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, video_path.split("/")[-1])
    logger.info(f"video save_path is {save_path}")

    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    
    tracker = BYTETracker(args, frame_rate=30) # use the global variable args
    timer = Timer()
    frame_id = 0 # initialize the frame id
    results = []
    
    while True:
        
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time))) # 
        
        # if frame_id % 2 == 0:
            
            
            
        #     timer.toc() 
        #     break
        
        # Process the video
        # E.g. when drop the frame_id 2 then the ground truth of frame 2 should also be droped
        # 1. send the GT into this function
        # 2. send the method select and parameters into this function
        # 
        
        # Process the video
            
        ret_val, frame = cap.read()
        
        if ret_val: 
            outputs, img_info = predictor.inference(frame, timer) # maybe modify the model here
            if outputs[0] is not None:
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
                )
            else:
                timer.toc() # total_time += diff, calls + 1 update the average 
                online_im = img_info['raw_img']
     
            vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1


    res_file = osp.join(vis_folder, f"{timestamp}.txt")
    with open(res_file, 'w') as f:
        f.writelines(results)
    logger.info(f"save results to {res_file}")


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