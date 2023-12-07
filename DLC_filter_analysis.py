import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
import scipy as scipy
from scipy.signal import find_peaks
import matplotlib.image as mpimg
import DLC_functions as dlcfun

date = '20230512'
fly_number = '2'


path = f"C:/Users/ashsm/Documents/Stanford/bruker_deeplabcut/bruker_crop_training/bruker_crop-ash-2023-06-02/videos/{date}_fly{fly_number}DLC_resnet50_bruker_cropJun2shuffle1_100001.h5"

h5_savefile = f"C:/Users/ashsm/Documents/Stanford/bruker behavior/h5files_dlc/{date}_fly{fly_number}_dlc.h5"


##get data out of dlc h5 file
#the output is essentially x,y locations for each timestamp (and likelihoods)
#could just do a check to see if likelihoods are ok and then procede with analysis
#h5py file doesn't seem to have a header...but data is stored in ['df_with_missing']['table']

#could get header from csv or just input csv.It's not too big so its ok. Could restore in h5py with better organization

#ultimately look at xy for bodypart1 and bodypart2 (possibly rename in later labeling) 
# # and get eucl distance between them (do as array slicing like foraging code)
#bodyparts	bodypart1	bodypart1	bodypart1	bodypart2	bodypart2	bodypart2	bodypart3	bodypart3	bodypart3	objectA	objectA	objectA
#coords	x	y	likelihood	x	y	likelihood	x	y	likelihood	x	y	likelihood

# with h5py.File(path, 'r') as f:
#     print(f.keys())
#     index_i = np.array(f['df_with_missing']['_i_table']['index'])
#     abounds = np.array(f['df_with_missing']['_i_table']['index']['abounds'])
#     data = np.array(f['df_with_missing']['table'])

# print('hello', data[:][1])
# print(data.dtype)
# print(data[:]['values_block_0'][:,0]) ##this is the first column (head x)

def get_data_column(path, index):
    """takes path to h5 file and returns the column of data for specific index --- 
    must know the index for corresponding labeled data-may need to check csv because no header in h5 file"""
    with h5py.File(path, 'r') as f:
        data = np.array(f['df_with_missing']['table'])

        #this data is formatted weird. values block indicates the psuedo dict key toget to data part, 
        # then want every column
        specified_column = data[:]['values_block_0'][:,index]
        return specified_column
    
### the labels for these will change if the config file changes. 
# scorer	DLC_resnet50_bruker_cropJun2shuffle1_10001	DLC_resnet50_bruker_cropJun2shuffle1_10001	DLC_resnet50_bruker_cropJun2shuffle1_10001	DLC_resnet50_bruker_cropJun2shuffle1_10001	DLC_resnet50_bruker_cropJun2shuffle1_10001	DLC_resnet50_bruker_cropJun2shuffle1_10001	DLC_resnet50_bruker_cropJun2shuffle1_10001	DLC_resnet50_bruker_cropJun2shuffle1_10001	DLC_resnet50_bruker_cropJun2shuffle1_10001	DLC_resnet50_bruker_cropJun2shuffle1_10001	DLC_resnet50_bruker_cropJun2shuffle1_10001	DLC_resnet50_bruker_cropJun2shuffle1_10001
# bodyparts	head	head	head	proboscis_tip	proboscis_tip	proboscis_tip	extended	extended	extended	neck	neck	neck

head_x = get_data_column(path, 0)
head_y = get_data_column(path, 1)
head_likelihood = get_data_column(path, 2)
proboscis_x = get_data_column(path, 3)
proboscis_y = get_data_column(path, 4)
proboscis_likelihood = get_data_column(path, 5)
extend_x = get_data_column(path, 6)
extend_y = get_data_column(path, 7)
extend_likelihood = get_data_column(path, 8)
neck_x = get_data_column(path, 9)
neck_y = get_data_column(path, 10)
neck_likelihood = get_data_column(path, 11)

dlcfun.add_to_h5(h5_savefile, 'head_x', head_x)
dlcfun.add_to_h5(h5_savefile, 'head_y', head_y)
dlcfun.add_to_h5(h5_savefile, 'head_likelihood', head_likelihood)
dlcfun.add_to_h5(h5_savefile, 'proboscis_x', proboscis_x)
dlcfun.add_to_h5(h5_savefile, 'proboscis_y', proboscis_y)
dlcfun.add_to_h5(h5_savefile, 'proboscis_likelihood', proboscis_likelihood)
dlcfun.add_to_h5(h5_savefile, 'extend_x', extend_x)
dlcfun.add_to_h5(h5_savefile, 'extend_y', extend_y)
dlcfun.add_to_h5(h5_savefile, 'extend_likelihood', extend_likelihood)
dlcfun.add_to_h5(h5_savefile, 'neck_x', neck_x)
dlcfun.add_to_h5(h5_savefile, 'neck_y', neck_y)
dlcfun.add_to_h5(h5_savefile, 'neck_likelihood', neck_likelihood)

difference = np.sqrt((head_x - proboscis_x)**2 + (head_y - proboscis_y)**2)
neck_diff = np.sqrt((neck_x - proboscis_x)**2 + (neck_y - proboscis_y)**2)
ext_diff = np.sqrt((extend_x - proboscis_x)**2 + (extend_y - proboscis_y)**2)
dlcfun.add_to_h5(h5_savefile, 'difference', difference)
dlcfun.add_to_h5(h5_savefile, 'neck difference', neck_diff)
dlcfun.add_to_h5(h5_savefile, 'extension difference', ext_diff)

DLC_peaks, properties = scipy.signal.find_peaks(difference, height = 5, prominence = 5, distance = 30, width = 10)
dlcfun.add_to_h5(h5_savefile, 'DLC peaks unfiltered', DLC_peaks)

## make median filter of median_difference_PER (difference to median head position)
median_head_x = np.median(head_x)
median_head_y = np.median(head_y)
prominence = 5
width = 10
height = 40

median_difference_PER = np.sqrt((proboscis_x - np.ones(len(proboscis_x))*median_head_x)**2 + (proboscis_y - np.ones(len(proboscis_y))*median_head_y)**2)

filtered_median_difference_PER = scipy.signal.medfilt(median_difference_PER, 5)
filtered_DLC_peaks, filtered_properties = scipy.signal.find_peaks(filtered_median_difference_PER,   width = width, prominence = prominence)
print(len(filtered_DLC_peaks))

print(np.std(filtered_median_difference_PER))
filter_range = np.mean(filtered_median_difference_PER) + 1*np.std(filtered_median_difference_PER)
print(filter_range)


double_filtered_DLC_peaks = [peak for peak in filtered_DLC_peaks if filtered_median_difference_PER[peak] > filter_range]
print('double', len(double_filtered_DLC_peaks))

dlcfun.add_to_h5(h5_savefile, 'median filtered DLC peaks', filtered_DLC_peaks)
dlcfun.add_to_h5(h5_savefile, 'median and std filtered DLC peaks', double_filtered_DLC_peaks)