import torch
from cinc_challenge_functions.datareader import DataReader
from cinc_challenge_functions import transforms
import numpy as np


class Config:
    
    REMAP=False

    HASH_TABLE=DataReader.get_label_maps(path="cinc_challenge_functions/")
    SNOMED_TABLE = DataReader.read_table(path="cinc_challenge_functions/")
    SNOMED_24_ORDERD_LIST=list(HASH_TABLE[0].keys())
    
    
    output_sampling=32
    std=0.2
    
    TRANSFORM_DATA_TRAIN=transforms.Compose([
        transforms.Resample(output_sampling=output_sampling),
        ])
    
    
    TRANSFORM_LBL=transforms.SnomedToOneHot()
    
    
