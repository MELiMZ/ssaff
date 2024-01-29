import h5py, os, glob, cv2, re
import numpy as np
def wv3():
    #######################################################################################
    ##### WV3
    ms_lists = ['NIR1/','NIR2/','CoastalBlue/','RedEdge/','RGB/','Yellow/']
    #############################################
    # Train
    # 
    #.h5文件名
    t=h5py.File("../data/NewH5Data/WV3/train.h5","w")
    num_train = 3246    #train图片数目
    v=h5py.File("../data/NewH5Data/WV3/valid.h5","w")
    num_valid = 300    #valid图片数目

    t_lrms = t.create_dataset("lrms", (num_train, 8, 64, 64), 'i')
    v_lrms = v.create_dataset("lrms", (num_valid, 8, 64, 64), 'i')

    t_pan = t.create_dataset("pan", (num_train, 1, 256, 256), 'i')
    v_pan = v.create_dataset("pan", (num_valid, 1, 256, 256), 'i')

    t_upms = t.create_dataset("upms", (num_train, 8, 256, 256), 'i')
    v_upms = v.create_dataset("upms", (num_valid, 8, 256, 256), 'i')

    t_gt = t.create_dataset('gt', (num_train, 8, 256, 256), 'i')
    v_gt = v.create_dataset('gt', (num_valid, 8, 256, 256), 'i')

    t_srgb = t.create_dataset('srgb', (num_train, 3, 256, 256), 'i')
    v_srgb = v.create_dataset('srgb', (num_valid, 3, 256, 256), 'i')

    t_lrgb = t.create_dataset('lrgb', (num_train, 3, 64, 64), 'i')
    v_lrgb = v.create_dataset('lrgb', (num_valid, 3, 64, 64), 'i')

    img_path_file = '../data/data2017Add/WV3/train/'   # 图片目录
    index_n = 0
    for filename in os.listdir(img_path_file + 'LRMS/' + ms_lists[0]):
        if filename.endswith(".tif"):
            # lrms
            img_lrms = np.zeros((8, 64, 64), dtype=int)
            ic = 0
            for ms_list in ms_lists:            
                img_path = img_path_file + 'LRMS/' + ms_list + filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    lrgb = img.transpose(2,0,1)
                    img_lrms[ic,:,:] = img[:,:,0]
                    img_lrms[ic+1,:,:] = img[:,:,1]
                    img_lrms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_lrms[ic,:,:] = img
                    ic += 1
            if index_n < num_train:
                t_lrms[index_n] = img_lrms
                t_lrgb[index_n] = lrgb
            else:
                v_lrms[index_n - num_train] = img_lrms
                v_lrgb[index_n - num_train] = lrgb
            # upms
            img_upms = np.zeros((8, 256, 256), dtype=int)
            ic = 0
            for ms_list in ms_lists:
                img_path = img_path_file + 'UPMS/' + ms_list+filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    img_upms[ic,:,:] = img[:,:,0]
                    img_upms[ic+1,:,:] = img[:,:,1]
                    img_upms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_upms[ic,:,:] = img
                    ic += 1
            if index_n < num_train:
                t_upms[index_n] = img_upms
            else:
                v_upms[index_n - num_train] = img_upms
            # gt
            img_gt = np.zeros((8, 256, 256), dtype=int)
            ic = 0
            for ms_list in ms_lists:
                img_path = img_path_file + 'GT/' + ms_list+filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    img_gt[ic,:,:] = img[:,:,0]
                    img_gt[ic+1,:,:] = img[:,:,1]
                    img_gt[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_gt[ic,:,:] = img
                    ic += 1
            if index_n < num_train:
                t_gt[index_n] = img_gt
            else:
                v_gt[index_n - num_train] = img_gt
            # srgb
            img_srgb = np.zeros((3, 256, 256), dtype=int)
            img_path = img_path_file + 'SRGB/' + filename
            img = cv2.imread(img_path, 3)
            img_srgb[0,:,:] = img[:,:,0]
            img_srgb[1,:,:] = img[:,:,1]
            img_srgb[2,:,:] = img[:,:,2]
            if index_n < num_train:
                t_srgb[index_n] = img_srgb
            else:
                v_srgb[index_n - num_train] = img_srgb
            # pan
            img_pan = np.zeros((1, 256, 256), dtype=int)
            img_path = img_path_file + 'PAN/' + filename
            img = cv2.imread(img_path, 2)
            if index_n < num_train:
                t_pan[index_n] = img
            else:
                v_pan[index_n - num_train] = img
            
            index_n += 1

    #############################################
    # Test


    #.h5文件名
    t=h5py.File("../data/NewH5Data/WV3/test.h5","w")
    num_train = 100    #train图片数目

    t_lrms = t.create_dataset("lrms", (num_train, 8, 64, 64), 'i')
    t_pan = t.create_dataset("pan", (num_train, 1, 256, 256), 'i')
    t_upms = t.create_dataset("upms", (num_train, 8, 256, 256), 'i')
    t_gt = t.create_dataset('gt', (num_train, 8, 256, 256), 'i')
    t_srgb = t.create_dataset('srgb', (num_train, 3, 256, 256), 'i')
    t_lrgb = t.create_dataset('lrgb', (num_train, 3, 64, 64), 'i')

    img_path_file = '../data/data2017Add/WV3/test/'   # 图片目录
    index_n = 0
    for filename in os.listdir(img_path_file + 'LRMS/' + ms_lists[0]):
        if filename.endswith(".tif"):
            # lrms
            img_lrms = np.zeros((8, 64, 64), dtype=int)
            ic = 0
            for ms_list in ms_lists:            
                img_path = img_path_file + 'LRMS/' + ms_list + filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    lrgb = img.transpose(2,0,1)
                    img_lrms[ic,:,:] = img[:,:,0]
                    img_lrms[ic+1,:,:] = img[:,:,1]
                    img_lrms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_lrms[ic,:,:] = img
                    ic += 1
            t_lrms[index_n] = img_lrms
            t_lrgb[index_n] = lrgb
            # upms
            img_upms = np.zeros((8, 256, 256), dtype=int)
            ic = 0
            for ms_list in ms_lists:
                img_path = img_path_file + 'UPMS/' + ms_list+filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    img_upms[ic,:,:] = img[:,:,0]
                    img_upms[ic+1,:,:] = img[:,:,1]
                    img_upms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_upms[ic,:,:] = img
                    ic += 1
            t_upms[index_n] = img_upms
            # gt
            img_gt = np.zeros((8, 256, 256), dtype=int)
            ic = 0
            for ms_list in ms_lists:
                img_path = img_path_file + 'GT/' + ms_list+filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    img_gt[ic,:,:] = img[:,:,0]
                    img_gt[ic+1,:,:] = img[:,:,1]
                    img_gt[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_gt[ic,:,:] = img
                    ic += 1
            t_gt[index_n] = img_gt
            # srgb
            img_srgb = np.zeros((3, 256, 256), dtype=int)
            img_path = img_path_file + 'SRGB/' + filename
            img = cv2.imread(img_path, 3)
            img_srgb[0,:,:] = img[:,:,0]
            img_srgb[1,:,:] = img[:,:,1]
            img_srgb[2,:,:] = img[:,:,2]
            t_srgb[index_n] = img_srgb
            # pan
            img_pan = np.zeros((1, 256, 256), dtype=int)
            img_path = img_path_file + 'PAN/' + filename
            img = cv2.imread(img_path, 2)
            t_pan[index_n] = img
            
            index_n += 1


    #############################################
    # withoutRef


    #.h5文件名
    t=h5py.File("../data/NewH5Data/WV3/withoutRef.h5","w")
    num_train = 95    #train图片数目

    t_lrms = t.create_dataset("lrms", (num_train, 8, 200, 200), 'i')
    t_pan = t.create_dataset("pan", (num_train, 1, 800, 800), 'i')
    t_upms = t.create_dataset("upms", (num_train, 8, 800, 800), 'i')
    t_gt = t.create_dataset('gt', (num_train, 8, 800, 800), 'i')
    t_srgb = t.create_dataset('srgb', (num_train, 3, 800, 800), 'i')
    t_lrgb = t.create_dataset('lrgb', (num_train, 3, 200, 200), 'i')

    img_path_file = '../data/data2017Add/WV3/withoutRef/'   # 图片目录
    index_n = 0
    for filename in os.listdir(img_path_file + 'LRMS/' + ms_lists[0]):
        if filename.endswith(".tif"):
            # lrms
            img_lrms = np.zeros((8, 200, 200), dtype=int)
            ic = 0
            for ms_list in ms_lists:            
                img_path = img_path_file + 'LRMS/' + ms_list + filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    lrgb = img.transpose(2,0,1)
                    img_lrms[ic,:,:] = img[:,:,0]
                    img_lrms[ic+1,:,:] = img[:,:,1]
                    img_lrms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_lrms[ic,:,:] = img
                    ic += 1
            t_lrms[index_n] = img_lrms
            t_lrgb[index_n] = lrgb
            # upms
            img_upms = np.zeros((8, 800, 800), dtype=int)
            ic = 0
            for ms_list in ms_lists:
                img_path = img_path_file + 'UPMS/' + ms_list+filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    img_upms[ic,:,:] = img[:,:,0]
                    img_upms[ic+1,:,:] = img[:,:,1]
                    img_upms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_upms[ic,:,:] = img
                    ic += 1
            t_upms[index_n] = img_upms
            # srgb
            img_srgb = np.zeros((3, 800, 800), dtype=int)
            img_path = img_path_file + 'SRGB/' + filename
            img = cv2.imread(img_path, 3)
            img_srgb[0,:,:] = img[:,:,0]
            img_srgb[1,:,:] = img[:,:,1]
            img_srgb[2,:,:] = img[:,:,2]
            t_srgb[index_n] = img_srgb
            # pan
            img_pan = np.zeros((1, 800, 800), dtype=int)
            img_path = img_path_file + 'PAN/' + filename
            img = cv2.imread(img_path, 2)
            t_pan[index_n] = img
            
            index_n += 1




def pleiades():

    #######################################################################################
    ##### Pleiades
    #############################################
    ms_lists = ['NIR/','RGB/']
    # Train
    # 

    #.h5文件名
    t=h5py.File("../data/NewH5Data/Pleiades/train.h5","w")
    num_train = 5224    #train图片数目
    v=h5py.File("../data/NewH5Data/Pleiades/valid.h5","w")
    num_valid = 500    #valid图片数目

    t_lrms = t.create_dataset("lrms", (num_train, 4, 64, 64), 'i')
    v_lrms = v.create_dataset("lrms", (num_valid, 4, 64, 64), 'i')

    t_pan = t.create_dataset("pan", (num_train, 1, 256, 256), 'i')
    v_pan = v.create_dataset("pan", (num_valid, 1, 256, 256), 'i')

    t_upms = t.create_dataset("upms", (num_train, 4, 256, 256), 'i')
    v_upms = v.create_dataset("upms", (num_valid, 4, 256, 256), 'i')

    t_gt = t.create_dataset('gt', (num_train, 4, 256, 256), 'i')
    v_gt = v.create_dataset('gt', (num_valid, 4, 256, 256), 'i')

    t_srgb = t.create_dataset('srgb', (num_train, 3, 256, 256), 'i')
    v_srgb = v.create_dataset('srgb', (num_valid, 3, 256, 256), 'i')

    t_lrgb = t.create_dataset('lrgb', (num_train, 3, 64, 64), 'i')
    v_lrgb = v.create_dataset('lrgb', (num_valid, 3, 64, 64), 'i')

    img_path_file = '../data/data2017Add/Pleiades/train/'   # 图片目录
    index_n = 0
    for filename in os.listdir(img_path_file + 'LRMS/' + ms_lists[0]):
        if filename.endswith(".tif"):
            # lrms
            img_lrms = np.zeros((4, 64, 64), dtype=int)
            ic = 0
            for ms_list in ms_lists:            
                img_path = img_path_file + 'LRMS/' + ms_list + filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    lrgb = img.transpose(2,0,1)
                    img_lrms[ic,:,:] = img[:,:,0]
                    img_lrms[ic+1,:,:] = img[:,:,1]
                    img_lrms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_lrms[ic,:,:] = img
                    ic += 1
            if index_n < num_train:
                t_lrms[index_n] = img_lrms
                t_lrgb[index_n] = lrgb
            else:
                v_lrms[index_n - num_train] = img_lrms
                v_lrgb[index_n - num_train] = lrgb
            # upms
            img_upms = np.zeros((4, 256, 256), dtype=int)
            ic = 0
            for ms_list in ms_lists:
                img_path = img_path_file + 'UPMS/' + ms_list+filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    img_upms[ic,:,:] = img[:,:,0]
                    img_upms[ic+1,:,:] = img[:,:,1]
                    img_upms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_upms[ic,:,:] = img
                    ic += 1
            if index_n < num_train:
                t_upms[index_n] = img_upms
            else:
                v_upms[index_n - num_train] = img_upms
            # gt
            img_gt = np.zeros((4, 256, 256), dtype=int)
            ic = 0
            for ms_list in ms_lists:
                img_path = img_path_file + 'GT/' + ms_list+filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    img_gt[ic,:,:] = img[:,:,0]
                    img_gt[ic+1,:,:] = img[:,:,1]
                    img_gt[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_gt[ic,:,:] = img
                    ic += 1
            if index_n < num_train:
                t_gt[index_n] = img_gt
            else:
                v_gt[index_n - num_train] = img_gt
            # srgb
            img_srgb = np.zeros((3, 256, 256), dtype=int)
            img_path = img_path_file + 'SRGB/' + filename
            img = cv2.imread(img_path, 3)
            img_srgb[0,:,:] = img[:,:,0]
            img_srgb[1,:,:] = img[:,:,1]
            img_srgb[2,:,:] = img[:,:,2]
            if index_n < num_train:
                t_srgb[index_n] = img_srgb
            else:
                v_srgb[index_n - num_train] = img_srgb
            # pan
            img_pan = np.zeros((1, 256, 256), dtype=int)
            img_path = img_path_file + 'PAN/' + filename
            img = cv2.imread(img_path, 2)
            if index_n < num_train:
                t_pan[index_n] = img
            else:
                v_pan[index_n - num_train] = img
            
            index_n += 1
    # 
    #############################################
    # Test

    # 
    #.h5文件名
    t=h5py.File("../data/NewH5Data/Pleiades/test.h5","w")
    num_train = 180    #train图片数目

    t_lrms = t.create_dataset("lrms", (num_train, 4, 64, 64), 'i')
    t_pan = t.create_dataset("pan", (num_train, 1, 256, 256), 'i')
    t_upms = t.create_dataset("upms", (num_train, 4, 256, 256), 'i')
    t_gt = t.create_dataset('gt', (num_train, 4, 256, 256), 'i')
    t_srgb = t.create_dataset('srgb', (num_train, 3, 256, 256), 'i')
    t_lrgb = t.create_dataset('lrgb', (num_train, 3, 64, 64), 'i')

    img_path_file = '../data/data2017Add/Pleiades/test/'   # 图片目录
    index_n = 0
    for filename in os.listdir(img_path_file + 'LRMS/' + ms_lists[0]):
        if filename.endswith(".tif"):
            # lrms
            img_lrms = np.zeros((4, 64, 64), dtype=int)
            ic = 0
            for ms_list in ms_lists:            
                img_path = img_path_file + 'LRMS/' + ms_list + filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    lrgb = img.transpose(2,0,1)
                    img_lrms[ic,:,:] = img[:,:,0]
                    img_lrms[ic+1,:,:] = img[:,:,1]
                    img_lrms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_lrms[ic,:,:] = img
                    ic += 1
            t_lrms[index_n] = img_lrms
            t_lrgb[index_n] = lrgb
            # upms
            img_upms = np.zeros((4, 256, 256), dtype=int)
            ic = 0
            for ms_list in ms_lists:
                img_path = img_path_file + 'UPMS/' + ms_list+filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    img_upms[ic,:,:] = img[:,:,0]
                    img_upms[ic+1,:,:] = img[:,:,1]
                    img_upms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_upms[ic,:,:] = img
                    ic += 1
            t_upms[index_n] = img_upms
            # gt
            img_gt = np.zeros((4, 256, 256), dtype=int)
            ic = 0
            for ms_list in ms_lists:
                img_path = img_path_file + 'GT/' + ms_list+filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    img_gt[ic,:,:] = img[:,:,0]
                    img_gt[ic+1,:,:] = img[:,:,1]
                    img_gt[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_gt[ic,:,:] = img
                    ic += 1
            t_gt[index_n] = img_gt
            # srgb
            img_srgb = np.zeros((3, 256, 256), dtype=int)
            img_path = img_path_file + 'SRGB/' + filename
            img = cv2.imread(img_path, 3)
            img_srgb[0,:,:] = img[:,:,0]
            img_srgb[1,:,:] = img[:,:,1]
            img_srgb[2,:,:] = img[:,:,2]
            t_srgb[index_n] = img_srgb
            # pan
            img_pan = np.zeros((1, 256, 256), dtype=int)
            img_path = img_path_file + 'PAN/' + filename
            img = cv2.imread(img_path, 2)
            t_pan[index_n] = img
            
            index_n += 1
    # 

    #############################################
    # withoutRef

    # 
    #.h5文件名
    t=h5py.File("../data/NewH5Data/Pleiades/withoutRef.h5","w")
    num_train = 165    #train图片数目

    t_lrms = t.create_dataset("lrms", (num_train, 4, 200, 200), 'i')
    t_pan = t.create_dataset("pan", (num_train, 1, 800, 800), 'i')
    t_upms = t.create_dataset("upms", (num_train, 4, 800, 800), 'i')
    t_gt = t.create_dataset('gt', (num_train, 4, 800, 800), 'i')
    t_srgb = t.create_dataset('srgb', (num_train, 3, 800, 800), 'i')
    t_lrgb = t.create_dataset('lrgb', (num_train, 3, 200, 200), 'i')

    img_path_file = '../data/data2017Add/Pleiades/withoutRef/'   # 图片目录
    index_n = 0
    for filename in os.listdir(img_path_file + 'LRMS/' + ms_lists[0]):
        if filename.endswith(".tif"):
            # lrms
            img_lrms = np.zeros((4, 200, 200), dtype=int)
            ic = 0
            for ms_list in ms_lists:            
                img_path = img_path_file + 'LRMS/' + ms_list + filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    lrgb = img.transpose(2,0,1)
                    img_lrms[ic,:,:] = img[:,:,0]
                    img_lrms[ic+1,:,:] = img[:,:,1]
                    img_lrms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_lrms[ic,:,:] = img
                    ic += 1
            t_lrms[index_n] = img_lrms
            t_lrgb[index_n] = lrgb
            # upms
            img_upms = np.zeros((4, 800, 800), dtype=int)
            ic = 0
            for ms_list in ms_lists:
                img_path = img_path_file + 'UPMS/' + ms_list+filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    img_upms[ic,:,:] = img[:,:,0]
                    img_upms[ic+1,:,:] = img[:,:,1]
                    img_upms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_upms[ic,:,:] = img
                    ic += 1
            t_upms[index_n] = img_upms
            # # srgb
            img_srgb = np.zeros((3, 800, 800), dtype=int)
            img_path = img_path_file + 'SRGB/' + filename
            img = cv2.imread(img_path, 3)
            img_srgb[0,:,:] = img[:,:,0]
            img_srgb[1,:,:] = img[:,:,1]
            img_srgb[2,:,:] = img[:,:,2]
            t_srgb[index_n] = img_srgb
            # pan
            img_pan = np.zeros((1, 800, 800), dtype=int)
            img_path = img_path_file + 'PAN/' + filename
            img = cv2.imread(img_path, 2)
            t_pan[index_n] = img
            
            index_n += 1
    # 




def IKONOS():

    #######################################################################################
    ##### IKONOS
    #############################################
    ms_lists = ['NIR/','RGB/']
    # Train
    # 

    #.h5文件名
    t=h5py.File("../data/NewH5Data/IKONOS/train.h5","w")
    num_train = 4838    #train图片数目
    v=h5py.File("../data/NewH5Data/IKONOS/valid.h5","w")
    num_valid = 400    #valid图片数目

    t_lrms = t.create_dataset("lrms", (num_train, 4, 64, 64), 'i')
    v_lrms = v.create_dataset("lrms", (num_valid, 4, 64, 64), 'i')

    t_pan = t.create_dataset("pan", (num_train, 1, 256, 256), 'i')
    v_pan = v.create_dataset("pan", (num_valid, 1, 256, 256), 'i')

    t_upms = t.create_dataset("upms", (num_train, 4, 256, 256), 'i')
    v_upms = v.create_dataset("upms", (num_valid, 4, 256, 256), 'i')

    t_gt = t.create_dataset('gt', (num_train, 4, 256, 256), 'i')
    v_gt = v.create_dataset('gt', (num_valid, 4, 256, 256), 'i')

    t_srgb = t.create_dataset('srgb', (num_train, 3, 256, 256), 'i')
    v_srgb = v.create_dataset('srgb', (num_valid, 3, 256, 256), 'i')

    t_lrgb = t.create_dataset('lrgb', (num_train, 3, 64, 64), 'i')
    v_lrgb = v.create_dataset('lrgb', (num_valid, 3, 64, 64), 'i')

    img_path_file = '../data/data2017Add/IKONOS/train/'   # 图片目录
    index_n = 0
    for filename in os.listdir(img_path_file + 'LRMS/' + ms_lists[0]):
        if filename.endswith(".tif"):
            # lrms
            img_lrms = np.zeros((4, 64, 64), dtype=int)
            ic = 0
            for ms_list in ms_lists:            
                img_path = img_path_file + 'LRMS/' + ms_list + filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    lrgb = img.transpose(2,0,1)
                    img_lrms[ic,:,:] = img[:,:,0]
                    img_lrms[ic+1,:,:] = img[:,:,1]
                    img_lrms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_lrms[ic,:,:] = img
                    ic += 1
            if index_n < num_train:
                t_lrms[index_n] = img_lrms
                t_lrgb[index_n] = lrgb
            else:
                v_lrms[index_n - num_train] = img_lrms
                v_lrgb[index_n - num_train] = lrgb
            # upms
            img_upms = np.zeros((4, 256, 256), dtype=int)
            ic = 0
            for ms_list in ms_lists:
                img_path = img_path_file + 'UPMS/' + ms_list+filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    img_upms[ic,:,:] = img[:,:,0]
                    img_upms[ic+1,:,:] = img[:,:,1]
                    img_upms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_upms[ic,:,:] = img
                    ic += 1
            if index_n < num_train:
                t_upms[index_n] = img_upms
            else:
                v_upms[index_n - num_train] = img_upms
            # gt
            img_gt = np.zeros((4, 256, 256), dtype=int)
            ic = 0
            for ms_list in ms_lists:
                img_path = img_path_file + 'GT/' + ms_list+filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    img_gt[ic,:,:] = img[:,:,0]
                    img_gt[ic+1,:,:] = img[:,:,1]
                    img_gt[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_gt[ic,:,:] = img
                    ic += 1
            if index_n < num_train:
                t_gt[index_n] = img_gt
            else:
                v_gt[index_n - num_train] = img_gt
            # srgb
            img_srgb = np.zeros((3, 256, 256), dtype=int)
            img_path = img_path_file + 'SRGB/' + filename
            img = cv2.imread(img_path, 3)
            img_srgb[0,:,:] = img[:,:,0]
            img_srgb[1,:,:] = img[:,:,1]
            img_srgb[2,:,:] = img[:,:,2]
            if index_n < num_train:
                t_srgb[index_n] = img_srgb
            else:
                v_srgb[index_n - num_train] = img_srgb
            # pan
            img_pan = np.zeros((1, 256, 256), dtype=int)
            img_path = img_path_file + 'PAN/' + filename
            img = cv2.imread(img_path, 2)
            if index_n < num_train:
                t_pan[index_n] = img
            else:
                v_pan[index_n - num_train] = img
            
            index_n += 1
    # 
    #############################################
    # Test

    # 
    #.h5文件名
    t=h5py.File("../data/NewH5Data/IKONOS/test.h5","w")
    num_train = 100    #train图片数目

    t_lrms = t.create_dataset("lrms", (num_train, 4, 64, 64), 'i')
    t_pan = t.create_dataset("pan", (num_train, 1, 256, 256), 'i')
    t_upms = t.create_dataset("upms", (num_train, 4, 256, 256), 'i')
    t_gt = t.create_dataset('gt', (num_train, 4, 256, 256), 'i')
    t_srgb = t.create_dataset('srgb', (num_train, 3, 256, 256), 'i')
    t_lrgb = t.create_dataset('lrgb', (num_train, 3, 64, 64), 'i')

    img_path_file = '../data/data2017Add/IKONOS/test/'   # 图片目录
    index_n = 0
    for filename in os.listdir(img_path_file + 'LRMS/' + ms_lists[0]):
        if filename.endswith(".tif"):
            # lrms
            img_lrms = np.zeros((4, 64, 64), dtype=int)
            ic = 0
            for ms_list in ms_lists:            
                img_path = img_path_file + 'LRMS/' + ms_list + filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    lrgb = img.transpose(2,0,1)
                    img_lrms[ic,:,:] = img[:,:,0]
                    img_lrms[ic+1,:,:] = img[:,:,1]
                    img_lrms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_lrms[ic,:,:] = img
                    ic += 1
            t_lrms[index_n] = img_lrms
            t_lrgb[index_n] = lrgb
            # upms
            img_upms = np.zeros((4, 256, 256), dtype=int)
            ic = 0
            for ms_list in ms_lists:
                img_path = img_path_file + 'UPMS/' + ms_list+filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    img_upms[ic,:,:] = img[:,:,0]
                    img_upms[ic+1,:,:] = img[:,:,1]
                    img_upms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_upms[ic,:,:] = img
                    ic += 1
            t_upms[index_n] = img_upms
            # gt
            img_gt = np.zeros((4, 256, 256), dtype=int)
            ic = 0
            for ms_list in ms_lists:
                img_path = img_path_file + 'GT/' + ms_list+filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    img_gt[ic,:,:] = img[:,:,0]
                    img_gt[ic+1,:,:] = img[:,:,1]
                    img_gt[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_gt[ic,:,:] = img
                    ic += 1
            t_gt[index_n] = img_gt
            # srgb
            img_srgb = np.zeros((3, 256, 256), dtype=int)
            img_path = img_path_file + 'SRGB/' + filename
            img = cv2.imread(img_path, 3)
            img_srgb[0,:,:] = img[:,:,0]
            img_srgb[1,:,:] = img[:,:,1]
            img_srgb[2,:,:] = img[:,:,2]
            t_srgb[index_n] = img_srgb
            # pan
            img_pan = np.zeros((1, 256, 256), dtype=int)
            img_path = img_path_file + 'PAN/' + filename
            img = cv2.imread(img_path, 2)
            t_pan[index_n] = img
            
            index_n += 1
    # 

    #############################################
    # withoutRef

    # 
    #.h5文件名
    t=h5py.File("../data/NewH5Data/IKONOS/withoutRef.h5","w")
    num_train = 95    #train图片数目

    t_lrms = t.create_dataset("lrms", (num_train, 4, 200, 200), 'i')
    t_pan = t.create_dataset("pan", (num_train, 1, 800, 800), 'i')
    t_upms = t.create_dataset("upms", (num_train, 4, 800, 800), 'i')
    t_gt = t.create_dataset('gt', (num_train, 4, 800, 800), 'i')
    t_srgb = t.create_dataset('srgb', (num_train, 3, 800, 800), 'i')
    t_lrgb = t.create_dataset('lrgb', (num_train, 3, 200, 200), 'i')

    img_path_file = '../data/data2017Add/IKONOS/withoutRef/'   # 图片目录
    index_n = 0
    for filename in os.listdir(img_path_file + 'LRMS/' + ms_lists[0]):
        if filename.endswith(".tif"):
            # lrms
            img_lrms = np.zeros((4, 200, 200), dtype=int)
            ic = 0
            for ms_list in ms_lists:            
                img_path = img_path_file + 'LRMS/' + ms_list + filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    lrgb = img.transpose(2,0,1)
                    img_lrms[ic,:,:] = img[:,:,0]
                    img_lrms[ic+1,:,:] = img[:,:,1]
                    img_lrms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_lrms[ic,:,:] = img
                    ic += 1
            t_lrms[index_n] = img_lrms
            t_lrgb[index_n] = lrgb
            # upms
            img_upms = np.zeros((4, 800, 800), dtype=int)
            ic = 0
            for ms_list in ms_lists:
                img_path = img_path_file + 'UPMS/' + ms_list+filename
                if ms_list == 'RGB/':
                    img = cv2.imread(img_path, 3)
                    img_upms[ic,:,:] = img[:,:,0]
                    img_upms[ic+1,:,:] = img[:,:,1]
                    img_upms[ic+2,:,:] = img[:,:,2]
                    ic += 3
                else:
                    img = cv2.imread(img_path, 2)
                    img_upms[ic,:,:] = img
                    ic += 1
            t_upms[index_n] = img_upms
            # # srgb
            img_srgb = np.zeros((3, 800, 800), dtype=int)
            img_path = img_path_file + 'SRGB/' + filename
            img = cv2.imread(img_path, 3)
            img_srgb[0,:,:] = img[:,:,0]
            img_srgb[1,:,:] = img[:,:,1]
            img_srgb[2,:,:] = img[:,:,2]
            t_srgb[index_n] = img_srgb
            # pan
            img_pan = np.zeros((1, 800, 800), dtype=int)
            img_path = img_path_file + 'PAN/' + filename
            img = cv2.imread(img_path, 2)
            t_pan[index_n] = img
            
            index_n += 1
    # 

IKONOS()