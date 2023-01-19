import cv2 as cv
import numpy as np
import glob
import time
import tqdm
import pandas as pd
import numpy as np
from os import path

class VideoPreprocessor():

    def __init__(self):
        pass
        

    

    def read_video_info(self,video_path):
        '''
        Read the video's frame, size, FPS
        '''
        cap = cv.VideoCapture(video_path)
        frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv.CAP_PROP_FPS)
        duration = frames/fps
        print('In the video',video_path+':')
        print('Frames:',int(frames))
        print('Size:',int(width),'*',int(height))
        print('FPS:',fps)
        return frames,(width,height),fps,duration


    def frame2video(frame_path,file_name=None):

        if file_name == None:
            file_name = frame_path+'.mp4'

        frames = glob.glob(frame_path+'/*.jpg')

        img0 = cv.imread(frames[0])


        height,width = img0.shape[:2]

        video_writer = cv.VideoWriter(file_name, fourcc=cv.VideoWriter_fourcc(*"mp4v"), fps=30, frameSize=(width, height), isColor=True)

        for frame in tqdm.tqdm(frames):

            img = cv.imread(frame)

            video_writer.write(img)

        video_writer.release()

        print('Saved video as',file_name)

        return file_name


    def process_video(self,inp,desired_framerate,format  = 'webm'):
        finame = inp.split('\\')[-1]
       
        file = f'{inp}\{finame}-raw.{format}'
        if not path.exists(file):
            print('Could not find video file')
            return None,None
        
        gt_file = f'{inp}\\gt\\gt.txt'
        found_gt = True
        if not path.exists(gt_file):
            print('Could not find Ground truth')
            found_gt = False
        if(found_gt):
            gt_df = self.read_gt(gt_file)
            desired_gt_df = pd.DataFrame(columns=gt_df.columns)

        cap = cv.VideoCapture(file)
        original_frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        original_fps = cap.get(cv.CAP_PROP_FPS)
        duration = original_frame_count/original_fps

        desired_fps = desired_framerate
        desired_frame_count = duration * desired_fps
        fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
        output_flie = f'{inp}\{finame}_processed.mp4'
        out = cv.VideoWriter(output_flie,fourcc,desired_fps,(int(width),  int(height)),1)

        frames_to_store = []
        f_count = 1
        desired_f_count = 1
        mod_sum = 0
        fps_diff = original_fps/desired_fps
        gts = []
        while cap.isOpened():
            ret, frame  = cap.read()

            if not ret:
                break
            
            if mod_sum < 1:
                out.write(frame)
                frames_to_store.append(f_count)
                if(found_gt):
                    gt = gt_df.loc[gt_df['frame_id']==f_count].copy()
                    gt['frame_id'] = desired_f_count
                    gts.append(gt)
                mod_sum += fps_diff - 1
                desired_f_count += 1
                '''while mod_sum < 0:
                    out.write(frame)
                    frames_to_store.append(f_count)'''
              
            else:
                mod_sum -= 1
            
            f_count += 1

        cap.release()
        out.release()
        
        print(f'Reduced the FPS from {original_fps} to {desired_fps}')
        frames_to_remove = [x for x in range(f_count) if x not in frames_to_store]
        gt_out_file = None
        if(found_gt):
            desired_gt_df = pd.concat(gts)
            gt_out_file = inp + '\\gt\\gt_processed.txt'
            desired_gt_df.to_csv(gt_out_file +'_processed.txt',header=0,index=0)
        return (output_flie,gt_out_file)


    def process_folder(self,folder,desired_framerate,input_format = 'webm'):
        file_folders = glob.glob(f'{folder}*',recursive=True)
        
        pairs = {}
        for file in file_folders:
            video_output,gt_output = self.process_video(file,desired_framerate,input_format)
            if(video_output is not None):
                pairs[file] = {'video_output':video_output, 'ground_truth_output':gt_output}
        return pairs #Dictionary with{original file path : {video_output:processed file path
                     #                                      ground_truth_output : processed ground truth path}} key:value pairs.


    def process_gt(self,inp,frames_to_remove):
        
       
        file = f'{inp}\\gt\\gt.txt'
        
        if not path.exists(file):
            print('Could not find Ground truth')
            return None

        df = self.read_gt(file)
        df = df.loc[df['frame_id'].apply(lambda x: x not in frames_to_remove)]
        print(df)

    
    def read_gt(self,path):
        df = pd.read_csv(path,names=['frame_id','Trajectory_id','bbox_1','bbox_2','bbox_3','bbox_4','confidence','class','vis_ratio'])    
        # print('In ground Truth')
        # print('Trajectories:',df['Trajectory_id'].values.max())
        # print('Bounding boxes:',df.index.max())
        return df
        
  
        
   


if __name__ == '__main__':
    vp = VideoPreprocessor()

    processed_files = vp.process_folder('MOT17\\test\\',15)
    file_out = vp.process_video('MOT17\\test\\MOT17-03-SDP',10)
    for k,file in processed_files.items():
        print(k,':',file)