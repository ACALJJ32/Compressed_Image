import os
import shutil

input_path = 'results/swinir_classical_sr_x4'
save_path = 'select_frames'
os.makedirs(save_path, exist_ok=True)


for i in range(1, 16):
    clip = "{:03d}".format(i)
    clip_path = os.path.join(input_path, clip)
    frame_list = os.listdir(clip_path)

    curr_save_path = os.path.join(save_path, clip)
    os.makedirs(curr_save_path, exist_ok=True)

    for frame in frame_list:
        frame_index = os.path.basename(frame).split('.')[0]
        frame_index = int(frame_index)
        if frame_index % 10 == 0:
            shutil.copy(os.path.join(clip_path, frame), os.path.join(curr_save_path, frame))