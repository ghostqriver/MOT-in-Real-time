import pandas as pd
import numpy as np


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
        self.df.to_csv(self.file_name.split('.')[0]+'_processed.txt',header=0,index=0)
        
    def rm_frame(self,frame_id):
        indexes = np.array(self.df.index)[self.df['frame_id']==frame_id]
        self.df.drop(indexes,inplace=True)
        print('Removed',len(indexes),'labels in frame',frame_id)