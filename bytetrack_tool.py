import re
import numpy as np
import matplotlib.pyplot as plt


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



def show_fps_mota(model_name,FPS_list,MOTA_list,fig_name):
    plt.figure(figsize=(8,5))
    plt.title(model_name)
    plt.plot(FPS_list[0],MOTA_list[0],marker='o',label='I.Simple frame reduction',c='blue')
    plt.plot(FPS_list[1],MOTA_list[1],marker='o',label='II.Static frame reduction',c='k')
    plt.vlines(30, plt.ylim()[0], plt.ylim()[1],linestyles='dashed',color='orange',)
    plt.xlabel('FPS')
    plt.ylabel('MOTA')
    plt.legend()
    plt.savefig(fig_name+'.jpg')