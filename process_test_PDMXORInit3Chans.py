
import numpy as np
import cv2 as cv
from Kmeans_checkInitial import KMeans
from encoding_PercepDecayMan import encoding
from encoding_PercepDecayMan3Channel import encoding_3Chanenel
from arguparse import args_parser
from sklearn.metrics import jaccard_score
from time import time
from experiment_log import PytorchExperimentLogger
import os
import random

if __name__ == '__main__':
    args = args_parser()
    np.random.seed(0)
    path = args.path
    dim = args.dim
    num_classes = args.numClass
    max_iterations = args.iterations
    changed_ratio = args.changed_ratio
    expLog = PytorchExperimentLogger("./log", "log", ShowTerminal=True)
    start = time()
    filelist = os.listdir(path)
    best_IoU = []
    dirlist = os.listdir(path)
    allBestIoU = []
    file_num = 0

    img_path = os.path.join(path, 'image/')
    gt_path = os.path.join(path, 'GT/')
    img_filelist = os.listdir(img_path)
    for imgFile in img_filelist:

        imgFile_name = imgFile.split(".")
        print(imgFile_name)
        if imgFile_name[0][-2:] == "GT" or 'DS_Store' == imgFile_name[1]:
            continue
        else:
            file_num += 1
            sample = cv.imread(img_path + imgFile)
            # print(img_path + imgFile)
            expLog.print("="*50)
            expLog.print(img_path + imgFile)
            expLog.print("changed_ratio: {}".format(changed_ratio))
            IMG_HEIGHT = sample.shape[0]
            IMG_WIDTH = sample.shape[1]
            sample1 = sample[:, :, 0:1].reshape(IMG_HEIGHT, IMG_WIDTH)
            sample2 = sample[:, :, 1:2].reshape(IMG_HEIGHT, IMG_WIDTH)
            sample3 = sample[:, :, 2:3].reshape(IMG_HEIGHT, IMG_WIDTH)
            expLog.print("sample1.shape {}".format(sample1.shape))
            print("sample1 == sample2", (sample1 == sample2).all())
            print("sample1 == sample3", (sample1 == sample3).all())

            if (sample1 == sample2).all() and (sample1 == sample3 ).all() and (sample3 == sample2).all():
                #continue
                print("1 channel")
                del sample2
                del sample3
                data = encoding(dim, 1, sample1, changed_ratio)
            else:
                print("3 channel")
                data = encoding_3Chanenel(dim,1,sample,changed_ratio)
                # expLog.print("data {}\n data shape: {}".format(data, data.shape))

            #sys.exit(0)
            random.seed(0)
            unique = np.unique(sample1)
            minColor = unique[0]
            maxColor = unique[-1]
            idx_minColor = np.argwhere(sample1 == minColor)
            idx_maxColor = np.argwhere(sample1 == maxColor)
            random_minColor = random.randint(0, len(idx_minColor)-1)
            random_maxColor = random.randint(0, len(idx_maxColor)-1)
            a1 = idx_minColor[random_minColor][0]
            b1 = idx_minColor[random_minColor][1]
            a2 = idx_maxColor[random_maxColor][0]
            b2 = idx_maxColor[random_maxColor][1]
            expLog.print("a1, b1:{}, {}, color:{}".format(a1, b1, sample1[a1][b1]))
            expLog.print("a2, b2:{}, {}, color:{}".format(a2, b2, sample1[a2][b2]))
            km2 = KMeans(data, num_classes, a1, b1, a2, b2)

            #km2 = KMeans(data, num_classes)
            centroids2, closest_centroids_ids2 = km2.train(max_iterations)
            expLog.print("closest_centroids_ids2.shape {}".format(closest_centroids_ids2.shape))
            closest_centroids_ids2 = closest_centroids_ids2.reshape(IMG_HEIGHT, IMG_WIDTH)
            expLog.print("closest_centroids_ids2.shape {}".format(closest_centroids_ids2.shape))
            closest_centroids_ids2 = closest_centroids_ids2.astype(np.uint8)
            expLog.print("mask {}".format(closest_centroids_ids2.shape))
            predict = []

            # In unsupervised segmentation, predicted labels are not aligned with GT labels.
            # Obtain foreground-background and background-foreground masks
            mask1 = closest_centroids_ids2.copy()
            mask1[mask1 != 0] = 1

            mask2 = mask1.copy()
            mask2[mask2 == 0] = 2
            mask2[mask2 == 1] = 0
            mask2[mask2 == 2] = 1

            predict.append(mask1)
            predict.append(mask2)

            GT = cv.imread(gt_path + imgFile, cv.IMREAD_GRAYSCALE)
            GT[GT == 255] = 1

            IoU = []

            for i in range(len(predict)):
                unique_GT = np.unique(GT)
                unique_pred = np.unique(predict[i])
                js = jaccard_score(GT.reshape(1, -1)[0], predict[i].reshape(1, -1)[0])
                IoU.append(js)
                expLog.print("js{}: {}".format(i+1, js))

            bestJs = max(IoU)  # for one image
            bestIdx = IoU.index(bestJs)
            expLog.print("best js{}: {}".format(bestIdx+1, bestJs))
            allBestIoU.append(bestJs)
            save_img = predict[bestIdx].copy()
            save_img[save_img == 1] = 255
            cv.imwrite("./bestMask/{}_GrayMask.png".format(imgFile_name[0]), save_img)


    expLog.print("="*30)
    expLog.print("all best IoU: {}".format(allBestIoU))
    allBestIoU = np.array(allBestIoU)
    avgIoU = np.mean(allBestIoU)
    expLog.print("avgIoU: {}".format(avgIoU))

    end = time()
    expLog.print("Time: {}".format(end - start))
