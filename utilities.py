#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Feb 2020
    Utilities script: functions used in local notebooks
@author: jesse
"""
##########
### IMPORTS
#########

# datacube stuff for DEA
import datacube

# plotting/imagery
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
from PIL import Image

# fio and maths
from glob import glob
import numpy as np
import csv

###########
### GLOBALS
###########

# how many white or black squares before we flag and remove?
# 5%
clip_frac = .05

# image name will be batchi/uid_product_time.png
imgname = "batch%1d/%04d_%s_%04d.png"
verbose=False

############
### METHODS
############

def jesse_normalise(imgarr, divby=4000.):
    img = np.copy(imgarr)
    img[img<0] = 0
    img = (img * (255./divby)).astype(np.float)
    img[img>255] = 255
    img = np.rint(img).astype(np.uint8) # round to integer
    return img

## CSV has list of points and dates to be pulled from sentinel
def get_batch(
    batch, 
    csvfile = '10k points v2.2.csv',
    x_space = -0.051,
    y_space = -0.034,
    ):
    """
    READ list of points and times (and unique identifiers) from csv
    ARGS:
        x_space: lon degrees to go from top left to bottom right
        y_space: lat degrees to go from top left to bottom right
    RETURNS:
        [[uid, time, x, y, batch],[...],...] 
    """
    uid_col = 0
    x_col, y_col = 1,2
    year_col, month_col = 3,4
    batch_col = 5
    list_of_chips = []
    # file has top-left corner, we add ~5km to get bottom corner

    with open(csvfile) as csv_file:
        readCSV = csv.reader(csv_file, delimiter=',')
        for row in readCSV:
            # 'batch' column splits our potential images for training/validation
            if row[batch_col]==str(batch):
                # top left to bottom right lat/lon bounds used to request data
                x_min = float(row[x_col])
                y_min = float(row[y_col])
                x_max = x_min-x_space
                y_max = y_min+y_space
                uid = int(row[uid_col])
                x=(x_min,x_max)
                y=(y_min,y_max)

                # request a month at a time
                year = int(row[year_col])
                month = int(row[month_col])
                date_str = "%4d-%02d"%(year,month)

                # save entries into a list
                list_of_chips.append([uid,date_str,x,y,batch])

    return list_of_chips

def landsat_grab(chip):
    """
        download and create image from landsat products
        INPUT:
            chip: [[id,'YYYY-MM',(lon0,lon1),(lat0,lat1),batch],[...]]
            dc: DataCube connection
    """
    dc = datacube.Datacube(app="landsat_grab")
    landsat_product = "ga_ls8c_ard_3"
    landsat_bands = ['nbart_red', 'nbart_green', 'nbart_blue']
    uid = chip[0]
    time = chip[1]
    x = chip[2]
    y = chip[3]
    batch=chip[4]
    if len(chip)>5:
        landsat_product=chip[5]
    res=(-25,25)
    if len(chip)>6:
        res = chip[6]

    ## Create a query object
    if verbose:
        print("INFO: chipper called:")
        print("INFO: x, y, time: ",x,y,time)
        
    query = {
        "x": x,
        "y": y,
        "time": time,
        "output_crs": "EPSG:3577",
        "resolution": res,
        "group_by": "solar_day",
    }
    
    ## download ard granule
    ds = dc.load(
        product=landsat_product,
        measurements = landsat_bands,
        **query
    )
    
    if verbose:
        print(ds)
    
    # Only get images if sentinel has some
    if len(ds.sizes)==0:
        print("WARNING: NO DATA FOR TIME RANGE",time)
    else:
        # We have data, iterate over available time steps
        for i in range(len(ds.time)):
            ## READ RGB into numpy array [y,x,channel]
            img = np.moveaxis(ds[['nbart_red', 'nbart_green', 'nbart_blue']].isel(time=i).to_array().values,0,2)
            if verbose:
                print("INFO: downloaded data details:")
                print("INFO: shape, type: ",img.shape, type(img))
                print("INFO: min, max: ", np.min(img), np.max(img))
            
            ## Normalise image to uint
            img = jesse_normalise(img)
            
            ## Check for too much white or black (probably cloud/missing)
            if np.sum(img==0) > clip_frac*np.size(img):
                continue
            if np.sum(img==255) > clip_frac*np.size(img):
                continue
            
            ## Cut to square
            n_y,n_x,_ = np.shape(img)
            dim_max=np.min([n_y,n_x])
            img_cut = img[:dim_max,:dim_max,:]
            if verbose:
                print("info: cut img size: ",img_cut.shape)
            
            ## Save figure
            # no longer saving flagged images
            fname=imgname%(batch,uid,landsat_product,i)
            imageio.imwrite(fname,img_cut)
            print("INFO: Saved figure: ",fname)


# BILINEAR seems to be the worst, so we use that

## Loop through images, add blurred image for each sentinel image

def resize_image(img, res=(160,160), interp = Image.BILINEAR):
    """
        resize image using PIL.Image.resize(res,method)
        could use ANTIALIAS, BICUBIC, NEAREST, BILINEAR
    """
    # open image
    return (Image.fromarray(img).resize(res, interp))
    

def compare_two_images(img1,img2):
    f,axes = plt.subplots(2,1,figsize=[13,14])
    
    for i,img in enumerate([img1,img2]):
        plt.sca(axes[i])
        plt.imshow(img)
    
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels("")
        ax.set_yticklabels("")
    plt.tight_layout()
    return f,axes

def compare_two_image_files(fname1,fname2):
    f,axes = plt.subplots(2,1,figsize=[13,14])
    
    for i,fname in enumerate([fname1,fname2]):
        plt.sca(axes[i])
        plt.imshow(mpimg.imread(fname))
        plt.title(fname)
    
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels("")
        ax.set_yticklabels("")
    plt.tight_layout()

def show_image(fname):

    plt.figure(figsize=[14,14])
    plt.imshow(mpimg.imread(fname))
    plt.xticks([],[])
    plt.yticks([],[])
    plt.ylabel("")
    plt.xlabel("")


def show_images(uid):
    fnames_s = glob("batch%d/%04d_sen*"%(batch,uid))
    fnames_l = glob("batch%d/%04d_lan*"%(batch,uid))
    # Show uid images in a grid
    f,axes = plt.subplots(2,3,figsize=[16,14])

    ## Loop over landsat images
    for i,fname in enumerate(fnames_l):
        plt.sca(axes[0,i])
        plt.imshow(mpimg.imread(fname))
        index = batch_1_full_list.index(fname)
        plt.title("%s (folder index:%d)"%(fname,index))

    ## loop over sentinel images
    for i,fname in enumerate(fnames_s):
        plt.sca(axes[1,i])
        plt.imshow(mpimg.imread(fname))
        index = batch_1_full_list.index(fname)
        plt.title("%s (folder index:%d)"%(fname,index))

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels("")
        ax.set_yticklabels("")
    plt.tight_layout()

