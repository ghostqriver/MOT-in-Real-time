import cv2

def read_video_info(video_path):
    '''
    Read the video's frame, size, FPS
    '''
    cap = cv2.VideoCapture(video_path)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('In the video',video_path+':')
    print('Frames:',int(frames))
    print('Size:',int(width),'*',int(height))
    print('FPS:',fps)