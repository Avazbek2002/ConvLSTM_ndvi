import os
import pandas as pd
from torch.utils.data import Dataset
import tifffile as tiff
from PIL import TiffImagePlugin
import numpy as np

TiffImagePlugin.OPEN = True

class NDVIDataset(Dataset):
    def __init__(self, dataroot, input_length = 4, output_length=1):
        self.datalist = []
        self.input_length = input_length
        time_step_list = sorted(os.listdir(dataroot))
        img_list = sorted(os.listdir(os.path.join(dataroot, time_step_list[0])))

        for img_i in range(len(img_list)):
            for i in range(len(time_step_list) - input_length - output_length):
                input_list = []
                for j in range(input_length):
                    j_time_step_path = os.path.join(dataroot, time_step_list[i+j])
                    j_time_step_img_path = os.path.join(j_time_step_path, img_list[img_i])
                    input_list.append(j_time_step_img_path)
                
                output_list = []
                for j in range(output_length):
                    i_output_path = os.path.join(dataroot, time_step_list[i+input_length+j])
                    i_output_image_path = os.path.join(i_output_path, img_list[img_i])
                    output_list.append(i_output_image_path)
                
                self.datalist.append((input_list, output_list))


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        in_imgs = []
        for img in self.datalist[idx][0]:
            single_image = np.transpose(tiff.imread(img), (2, 0, 1))
            in_imgs.append(single_image)
        
        input_imgs = np.array(in_imgs)

        out_imgs = []
        for img in self.datalist[idx][1]:
            single_image = np.transpose(tiff.imread(img), (2, 0, 1))
            out_imgs.append(single_image)
        
        output_imgs = np.array(out_imgs)

        return {"input": input_imgs, "output": output_imgs}