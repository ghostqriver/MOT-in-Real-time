import cv2 as cv
import numpy as np
import glob
import time

def read_video_info(video_path):
    '''
    Read the video's frame, size, FPS
    '''
    cap = cv.VideoCapture(video_path)
    frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv.CAP_PROP_FPS)
    print('In the video',video_path+':')
    print('Frames:',int(frames))
    print('Size:',int(width),'*',int(height))
    print('FPS:',fps)
    return frames,(width,height),fps








def reduce_video_framerate(input,desired_framerate,samelength):
    cap = cv.VideoCapture(input)
    #frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv.CAP_PROP_FPS)

    frame_rate = desired_framerate
    prev = 0
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    output_flie = f'{input.split(".")[0]}_processed.mp4'
    out = cv.VideoWriter(output_flie,fourcc,frame_rate,(int(width),  int(height)),1)


    while cap.isOpened():
        time_elapsed = time.time() - prev
        ret, frame  = cap.read()

        if not ret:
            break

        if time_elapsed > 1/frame_rate:
            prev = time.time()
            curr_frame = frame
            out.write(curr_frame)
            '''cv.imshow('frame', curr_frame)
            if cv.waitKey(1) == ord('q'):
                break'''
        elif(samelength):
            out.write(curr_frame)

    cap.release()
    out.release()
    #cv.destroyAllWindows()
    print(f'Reduced the FPS from {fps} to {desired_framerate}')
    return output_flie


def process_folder(folder,input_format,desired_framerate,samelength = True):
    files = glob.glob(f'{folder}*.{input_format}',recursive=True)
    #print(files)
    pairs = {}
    for file in files:
        output = reduce_video_framerate(file,desired_framerate,samelength)
        pairs[file] = output
    return pairs #Dictionary with{original file path : processed file path} key:value pairs.


if __name__ == '__main__':
    processed_files = process_folder('MOT17/test/','webm',15,False)

    for file in processed_files.values():
        read_video_info(file)