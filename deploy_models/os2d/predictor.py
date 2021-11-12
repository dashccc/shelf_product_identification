import sys
import json
import os
import warnings
import flask
import boto3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from os2d.modeling.model import build_os2d_from_config
from os2d.config import cfg
from os2d.structures.bounding_box import cat_boxlist, BoxList, boxlist_iou
import  os2d.utils.visualization as visualizer
from os2d.structures.feature_map import FeatureMapSize
from os2d.utils import setup_logger, read_image, get_image_size_after_resize_preserving_aspect_ratio
logger = setup_logger("OS2D")

app = flask.Flask(__name__)

s3_client = boto3.client('s3')

def filtered_bboxes(boxes, labels, scores, score_threshold=0.0, max_dets=None):
    good_ids = torch.nonzero(scores.float() > score_threshold).view(-1)
    if good_ids.numel() > 0:
        if max_dets is not None:
                _, ids = scores[good_ids].sort(descending=True)
                good_ids = good_ids[ids[-max_dets:]]
                # print(good_ids)
        boxes = boxes[good_ids].cpu()
        labels = labels[good_ids].cpu()
        scores = scores[good_ids].cpu()
        label_names = [ "Cl "+ str(l.item()) for l in labels]
        box_colors = ["yellow"] * len(boxes)
    else:
        boxes = BoxList.create_empty(boxes.image_size)
        labels = torch.LongTensor(0)
        scores = torch.FloatTensor(0)
        label_names = []
        box_colors = []
    # boxes = boxes.bbox_xyxy
    return boxes

#prepare model
cfg.is_cuda = torch.cuda.is_available()
cfg.init.model = "/opt/ml/code/os2d_v2-train.pth"
net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)
net.eval()

#coding classes imgs
classes=os.listdir('data/demo/classes')
imgs=[]
for i in classes:
    if i[-4:]=='.jpg':
        imgs.append(i)
print(imgs)
class_images = [read_image("data/demo/classes/{}".format(imgs[i])) for i in range(len(imgs))]
class_ids = list(range(len(imgs)))
transform_image = transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(img_normalization["mean"], img_normalization["std"])
                      ])
class_images_th = []
for class_image in class_images:
    h, w = get_image_size_after_resize_preserving_aspect_ratio(h=class_image.size[1],
                                                               w=class_image.size[0],
                                                               target_size=cfg.model.class_image_size)
    class_image = class_image.resize((w, h))

    class_image_th = transform_image(class_image)
    if cfg.is_cuda:
        class_image_th = class_image_th.cuda()

    class_images_th.append(class_image_th)

@torch.no_grad()
def inference_classes(class_images_th, net):
    loc_prediction=[]
    class_prediction=[]
    class_conv_layer_batched=[]
    transform_corners=[]
    for i in range(len(class_images_th)):
        class_feature_maps = net.net_label_features([class_images_th[i]])
        class_head=net.os2d_head_creator.create_os2d_head(class_feature_maps)
        class_conv_layer_batched.append(class_head)
    return class_conv_layer_batched
class_conv_layer_batched=inference_classes(class_images_th, net)

#detect
@torch.no_grad()
def detect(source,class_conv_layer_batched):
    loc_prediction=[]
    class_prediction=[]
    class_conv_layer_batched=[]
    transform_corners=[]
    input_image = read_image(source)
    h, w = get_image_size_after_resize_preserving_aspect_ratio(h=input_image.size[1],
                                                               w=input_image.size[0],
                                                               target_size=4040)
    input_image = input_image.resize((w, h))
    input_image_th = transform_image(input_image)
    input_image_th = input_image_th.unsqueeze(0)
    if cfg.is_cuda:
        input_image_th = input_image_th.cuda()
    feature_map = net.net_feature_maps(input_image_th)
    for i in range(len(class_conv_layer_batched)):
        loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = net(feature_maps=feature_map, class_head=class_conv_layer_batched[i])
        loc_prediction.append(loc_prediction_batch[0])
        class_prediction.append(class_prediction_batch[0])
        transform_corners.append(transform_corners_batch[0])
    loc_prediction,class_prediction,transform_corners=torch.cat(loc_prediction,dim=0),torch.cat(class_prediction,dim=0),torch.cat(transform_corners,dim=0)
    image_loc_scores_pyramid = [loc_prediction]
    image_class_scores_pyramid = [class_prediction]
    img_size_pyramid = [FeatureMapSize(img=input_image_th)]
    transform_corners_pyramid = [transform_corners]
    boxes = box_coder.decode_pyramid(image_loc_scores_pyramid, image_class_scores_pyramid,
                                           img_size_pyramid, class_ids,
                                           nms_iou_threshold=cfg.eval.nms_iou_threshold,
                                           nms_score_threshold=cfg.eval.nms_score_threshold,
                                           transform_corners_pyramid=transform_corners_pyramid)                                 
    boxes.remove_field("default_boxes") 
    labels = boxes.get_field("labels").clone()
    scores = boxes.get_field("scores").clone()
    b=filtered_bboxes(boxes,labels, scores, score_threshold=float("0.55"), max_dets=100)
    iou=boxlist_iou(b,b)
    idx=torch.nonzero(torch.sum(iou,dim=0)<1.5).view(-1)
    b = b[idx]
    s=np.array(b.get_field('scores').cpu())
    l=np.array(b.get_field('labels').cpu())
    s=np.expand_dims(s,axis=1)
    l=np.expand_dims(l,axis=1)
    xyxy=np.array(b.bbox_xyxy.cpu())
    res=np.concatenate((l,s,xyxy),axis=1)
    return res
    
@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    # print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")

    #parse json in request
    print ("<<<< flask.request.content_type", flask.request.content_type)

    if flask.request.content_type == 'application/x-image':
        image_as_bytes = io.BytesIO(flask.request.data)
        img = Image.open(image_as_bytes)
        download_file_name = '/tmp/tmp.jpg'
        img.save(download_file_name)
        print ("<<<<download_file_name ", download_file_name)
    else:
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)

        bucket = data['bucket']
        image_uri = data['image_uri']

        download_file_name = '/tmp/'+image_uri.split('/')[-1]
        print ("<<<<download_file_name ", download_file_name)

        try:
            s3_client.download_file(bucket, image_uri, download_file_name)
        except:
            #local test
            download_file_name = './bus.jpg'

        print('Download finished!')

    inference_result = detect(download_file_name)
    
    _payload = json.dumps(inference_result,ensure_ascii=False)

    return flask.Response(response=_payload, status=200, mimetype='application/json')