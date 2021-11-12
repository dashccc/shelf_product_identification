import os
import numpy as np
import pandas as pd
import math


def gen_rel_matrix_by_rule(rule, cls):
    skus2num=dict()
    num2sku=dict()
    h=len(rule)
    max_skus=0
    numid=1
    for i in cls:
        skus2num[i]=numid
        num2sku[numid]=i
        numid+=1
    for i in range(h):
        max_skus=max(sum(rule[i].values()),max_skus)
        # for k,v in rule[i].items():
        #     skus2num[k]=numid
        #     num2sku[numid]=k
        #     numid+=1
    matrix=np.zeros((h,max_skus))
    for i in range(h):
        flag=0
        for k, v in rule[i].items():
            matrix[i][flag:flag+v]=skus2num[k]
            flag+=v
    return matrix, skus2num, num2sku

def row_clustering(boxes):
    boxes_row = []
    flags = np.zeros((boxes.shape[0]), dtype=int)
    # flags=np.array(flags)
    # print(flags)
    i = 0
    while (flags == 0).any() and i < len(boxes):
        if flags[i] == 1:
            i += 1
            continue
        flags[i] = 1
        row = []
        flag = boxes[i]
        row.append(flag)
        for j in range(boxes.shape[0]):
            if flags[j] == 1:
                continue
            else:
                if boxes[j][1]<flag[1]:
                    A=boxes[j]
                    B=flag
                else:
                    A=flag
                    B=boxes[j]
                if A[3]-B[1]<=0:
                    continue
                else:
                    overlap = A[3]-B[1]
                    dm = min(abs(A[1]-A[3]),abs(B[1]-B[3]))
                    if overlap/dm>=0.8:
                        row.append(boxes[j])
                        flags[j] = 1
                    else:
                        continue
        row.sort(key=lambda x: x[0])
        boxes_row.append(row)
        i += 1
        boxes_row.sort(key=lambda x:x[0][1])
    return boxes_row

def get_bbox_dist(bboxes):
    # 输入是完成行聚类后的bbox,形状是[rows,cols]
    # 输出是list，形状是[rows,cols],每一行的每个元素代表该位置前后bbox的距离
    Res=[]
    Mean_weight=0
    min_x=100000
    for i in range(len(bboxes)):
        min_x=min(min_x,bboxes[i][0][0])
    for i in range(len(bboxes)):
        flag=0
        res=[]
        mean_weight=0
        for j in range(len(bboxes[i])):
            mean_weight+=abs(bboxes[i][j][0]-bboxes[i][j][2])/len(bboxes[i])
            if j ==0:
                res.append(max(0,bboxes[i][j][0]-min_x))
            else:
                dist=max(0,bboxes[i][j][0]-bboxes[i][flag][2])
                res.append(dist)
                flag+=1
        Mean_weight+=mean_weight/len(bboxes)
        Res.append(res)
    return Res,Mean_weight
def gen_rel_matrix_by_detect(res,skus2num):
    rows=row_clustering(res)
    print(rows)
    dist,Mean_weight=get_bbox_dist(rows)
    print('dist',dist)
    h=len(rows)
    w=0
    for i in range(h):
        w=max(w,len(rows[i]))
    matrix=np.zeros((h,w))
    for i in range(h):
        j=0
        idx_row=0
        idx_dist=0
        while (j<len(rows[i]) or idx_row<len(rows[i])):
            print(idx_dist)
        # for j in range(len(rows[i])):
            if  dist[i][idx_dist]>=0.8*Mean_weight:#idx_dist<len(dist[i]) and
                num=int(dist[i][idx_dist]//(0.8*Mean_weight))
                matrix[i][j:j+num]=0
                matrix[i][j+num]=skus2num[str(int(rows[i][idx_row][-1]))]
                idx_row+=1
                j+=num+1
                idx_dist+=1
            else:
                matrix[i][j]=skus2num[str(int(rows[i][idx_row][-1]))]
                print("i, j:", i, j)
                idx_row+=1
                j+=1
                idx_dist+=1
    return matrix,rows

def relative_matching(cls, rule, res):
    error=[]
    true=[]
    matrix1, skus2num, num2sku=gen_rel_matrix_by_rule(rule, cls)
    matrix2, rows=gen_rel_matrix_by_detect(res,skus2num)
    print(matrix1.shape)
    print(matrix2.shape)
    if matrix1.shape!=matrix2.shape:
        if matrix1.shape[1]>=matrix2.shape[1]:
            matrix1=matrix1[:matrix2.shape[0],:matrix2.shape[1]]
        else:
            matrix2=matrix2[:matrix1.shape[0],:matrix1.shape[1]]
    print(matrix1)
    print(matrix2)
    if (matrix1==matrix2).all():
        idx=matrix1!=matrix2
        list_of_coordinates_error=np.where(idx==True)
        list_of_coordinates_true=np.where(idx==False)
        for i in list(zip(list_of_coordinates_true[0],list_of_coordinates_true[1])):
            try:
                true.append(rows[i[0]][i[1]])
            except:
                continue
        return True,true, error
    else:
        idx=matrix1!=matrix2
        list_of_coordinates_error=np.where(idx==True)
        list_of_coordinates_true=np.where(idx==False)
        # print(list_of_coordinates2[0].shape)
        # print(list_of_coordinates2[1].shape)
        print(idx[list_of_coordinates_error])
        for i in list(zip(list_of_coordinates_error[0],list_of_coordinates_error[1])):
            try:
                error.append(rows[i[0]][i[1]])
            except:
                continue
        for i in list(zip(list_of_coordinates_true[0],list_of_coordinates_true[1])):
            try:
                true.append(rows[i[0]][i[1]])
            except:
                continue
        return False,true,error


if __name__ == '__main__':
    cls=['0','1','2','3','4','5','6','7','8']
    rule=[{'0':5,'2':6},
     {'3':6,'4':3,'5':6},
     {'6':6,'7':2,'8':6}]
    res=np.load('/Users/lvhaoran/AWScode/yolov5/res.npy')
    idx=res[:,-1]!=1.0
    res=res[idx]
    # print(res)
    print(res.shape)
    flag,true_bboxes,error_bboxes=relative_matching(cls, rule, res)
    print(flag)
    # print(error_bboxes)
    print(len(error_bboxes))
    # print(true_bboxes)
    print(len(true_bboxes))
    # (x1,y1,x2,y2,conf,cls)
    


