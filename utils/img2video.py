import cv2
import os 
from os.path import join as pjoin 

def path2video(folder, save_gif=True):
    
    lst = os.listdir(folder)
    lst = [i for i in lst if '.png' in i]
    lst.sort(key=lambda x: int(x.split('.')[0]))

    img_array = []
    for filename in lst:
        img = cv2.imread(pjoin(folder,filename))
        img_array.append(img)

    height, width, layers = img_array[0].shape
    size = (width, height)

    out = cv2.VideoWriter(pjoin(folder, 'video.mp4'), 0x7634706d, 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
    if save_gif:
        os.system('ffmpeg -i %s %s' % (pjoin(folder, 'video.mp4'), folder + '.gif'))
    
    os.system('rm -r ' + folder)

    return 

if __name__ == '__main__':
    
    test_folder = '../logs/video/grasp_cube_state_ppo/debug_seed2166/Iter0'
    path2video(test_folder, save_gif=True)
