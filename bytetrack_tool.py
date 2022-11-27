import re
import numpy as np

def fps_from_log(log_path='log.txt'):
  '''
  Extract the avg fps from the bytetrack's log file
  Please modify it or use your own FPS calculation funtion when the output file format is not suitable
  '''
  log_path='log.txt'
  with open(log_path, 'r') as file:
    text = file.read().replace('\n', '')
  log_fps_s = re.findall(r'\(\d+\.\d+\sfps\)', text)
  log_fps_s = list(map(lambda x : float(re.findall('\d+.\d+',x)[0]),log_fps_s))

  avg_fps = sum(log_fps_s[1:])/len(log_fps_s[1:])
  print('The average fps is',avg_fps)
  return avg_fps 