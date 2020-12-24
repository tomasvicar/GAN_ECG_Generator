import os
import glob
import numpy as np
import torch

from cinc_challenge_functions.datareader import DataReader
from cinc_challenge_functions import transforms
from cinc_challenge_functions.config import Config




input_directory = '../data_cinc2020_orig'
output_directory = '../data_cinc2020_sinusonly'
sample_length = Config.output_sampling * 8


file_list = glob.glob(input_directory + "/**/*.mat", recursive=True)



snomed_table = DataReader.read_table(path="cinc_challenge_functions/")
idx_mapping, label_mapping = DataReader.get_label_maps(path="cinc_challenge_functions/")
remap = Config.REMAP

counter = 0
for i,file in enumerate(file_list):
    print(i)
    
    sample_file_name = file
    header_file_name = file[:-3] + "hea"

    # Read data
    
    header = DataReader.read_header(header_file_name, snomed_table,remap=remap)

    sampling_frequency, resolution, age, sex, snomed_codes = header

    
    
    if ['426783006'] == snomed_codes:
        
        sample = DataReader.read_sample(sample_file_name)
        sample = Config.TRANSFORM_DATA_TRAIN(sample, input_sampling=sampling_frequency,gain=1/np.array(resolution))
        
        if sample.shape[1]>=sample_length:
        
        
            save_name = sample_file_name.replace('.mat','.npy').replace(input_directory,output_directory)
            
            head, tail = os.path.split(save_name)    
            
            if not os.path.exists(head):
                os.makedirs(head)
            
            
            
            sample = sample[:,:sample_length]
            
            np.save(save_name,sample)
            
        
            counter += 1
        
        
        
        
        
    
    
    








