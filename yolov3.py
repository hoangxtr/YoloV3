import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Activation, ZeroPadding2D, LeakyReLU, Input
from tensorflow.keras.models import Model
from utils import *
from config import *

def convolution(input_layer, filters, kernel_size, bn=True, activate=True, down_sample=False):
  if down_sample:
    padding='valid'
    strides=2
    input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
  else:
    padding ='same'
    strides=1
  x = Conv2D(
      filters, kernel_size, 
      padding=padding, strides=strides, 
      use_bias=not bn, 
      kernel_initializer=tf.random_normal_initializer(stddev=0.01), 
      bias_initializer=tf.constant_initializer(0.)
  )(input_layer)
  if bn:
    x = BatchNormalization()(x)
  if activate:
    x = LeakyReLU(alpha=0.1)(x)
  return x

def residual_block(input_layer, num_filters1, num_filters2):
  shorcut = input_layer
  x = convolution(shorcut, num_filters1, (1,1))
  x = convolution(x, num_filters2, (3,3))
  ret = shorcut + x
  return ret

def Darknet53(input_layer):
  '''
  Return 3 route of darknet: strides8, strides16, strides32
  '''
  x = convolution(input_layer, 32, (3,3))
  x = convolution(x, 64, (3,3), down_sample=True) # strides 2
  
  for i in range(1):
    x = residual_block(x, 32, 64)
  x = convolution(x, 128, (3,3), down_sample=True) # strides 4
  
  for i in range(2):
    x = residual_block(x, 64, 128)
  x = convolution(x, 256, (3,3), down_sample=True) # strides 8
  
  for i in range(8):
    x = residual_block(x, 128, 256)
  route1 = x
  x = convolution(x, 512, (3,3), down_sample=True)  # strides 16

  for i in range(8):
    x = residual_block(x, 256, 512)
  route2 = x
  x = convolution(x, 1024, (3,3), down_sample=True) # strides 32

  for i in range(4):
    x = residual_block(x, 512, 1024)
  return route1, route2, x

def upsample(input_layer):
  shape = input_layer.shape
  return tf.image.resize(input_layer, (shape[1]*2, shape[2]*2), method='nearest')
  



def YoloV3(input_layer, num_class):
  route1, route2, route3 = Darknet53(input_layer)
  
  # Build for route3:
  route3 = convolution(route3, 512, (1,1))
  route3 = convolution(route3, 1024, (3,3))
  route3 = convolution(route3, 512, (1,1))
  route3 = convolution(route3, 1024, (3,3))
  route3 = convolution(route3, 512, (1,1))
  
  lbranch = convolution(route3, 1024, (3,3))
  lbranch = convolution(lbranch, 3*(5+num_class), (1,1), bn=False, activate=False)

  route3 = convolution(route3, 256, (1,1))
  route3 = upsample(route3)
  route2 = tf.concat([route3, route2], axis=-1)
  route2 = convolution(route2, 256, (1,1))
  route2 = convolution(route2, 512, (3,3))
  route2 = convolution(route2, 256, (1,1))
  route2 = convolution(route2, 512, (3,3))
  route2 = convolution(route2, 256, (1,1))

  mbranch = convolution(route2, 512, (3,3))
  mbranch = convolution(mbranch, 3*(5+num_class), (1,1), bn=False, activate=False)

  route2 = convolution(route2, 128, (1,1))
  route2 = upsample(route2)
  route1 = tf.concat([route1, route2], axis=-1)
  route1 = convolution(route1, 128, (1,1))
  route1 = convolution(route1, 256, (3,3))
  route1 = convolution(route1, 128, (1,1))
  route1 = convolution(route1, 256, (3,3))
  route1 = convolution(route1, 128, (1,1))
  
  sbranch = convolution(route1, 256, (3,3))
  sbranch = convolution(sbranch, 3*(num_class+5), (1,1), bn=False, activate=False)
  return lbranch, mbranch, sbranch


def decode(conv_output, NUM_CLASS, i=0):
  # where i = 0, 1 or 2 to correspond to the three grid scales  
  conv_shape       = tf.shape(conv_output)
  batch_size       = conv_shape[0]
  output_size      = conv_shape[1]

  conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

  conv_raw_dxdy = conv_output[:, :, :, :, 0:2] # offset of center position     
  conv_raw_dwdh = conv_output[:, :, :, :, 2:4] # Prediction box length and width offset
  conv_raw_conf = conv_output[:, :, :, :, 4:5] # confidence of the prediction box
  conv_raw_prob = conv_output[:, :, :, :, 5: ] # category probability of the prediction box 

  # next need Draw the grid. Where output_size is equal to 13, 26 or 52  
  y = tf.range(output_size, dtype=tf.int32)
  y = tf.expand_dims(y, -1)
  y = tf.tile(y, [1, output_size])
  x = tf.range(output_size,dtype=tf.int32)
  x = tf.expand_dims(x, 0)
  x = tf.tile(x, [output_size, 1])

  xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
  xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
  xy_grid = tf.cast(xy_grid, tf.float32)

  # Calculate the center position of the prediction box:
  pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
  # Calculate the length and width of the prediction box:
  pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

  pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
  pred_conf = tf.sigmoid(conv_raw_conf) # object box calculates the predicted confidence
  pred_prob = tf.sigmoid(conv_raw_prob) # calculating the predicted probability category box object

  # calculating the predicted probability category box object
  return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def Create_yolov3(input_size, channels=3, num_class=3):
  input_layer = Input((input_size, input_size, channels))
  conv_tensors = YoloV3(input_layer, num_class)
  output_tensors = []
  for i, conv_tensor in enumerate(conv_tensors):
    pred_tensor = decode(conv_tensor, num_class, i)
    # output_tensors.append(conv_tensor)
    output_tensors.append(pred_tensor)
  ret = tf.keras.Model(input_layer, output_tensors)
  return ret

class YoloLoss:
  def __init__(self, lambda_coord, lambda_noobj, num_class, anchors, grid=13, ignore_thresh=0.5):
    self.lambda_coord = lambda_coord
    self.lambda_noobj = lambda_noobj
    self.num_class = num_class
    self.grid = grid
    self.anchors = anchors
    self.ignore_thresh = ignore_thresh
  
  def __call__(self, y_true, y_pred):
    '''
      y in shape (batch, grid, grid, 3, 5+num_class)
    '''
    pred_xy_rel = tf.sigmoid(y_pred[..., :2]) # (Batch, grid, grid, 3, 2)
    pred_wh_rel = y_pred[..., 2:4]  # (Batch, grid, grid, 3, 2)
    pred_obj = tf.sigmoid(y_pred[..., 4]) # (Batch, grid, grid, 3, 1)
    pred_class = tf.sigmoid(y_pred[..., 5:]) # (Batch, grid, grid, 3, num_class)

    true_xy_rel, true_wh_rel, true_obj, true_class = tf.split(y_true, [2,2,1,self.num_class], axis=-1)

    ignore_mask = self.calc_ignore_mask(y_pred, y_true)

    xy_loss = self.lambda_coord * tf.reduce_sum(tf.square(true_xy_rel-pred_xy_rel), axis=-1) \
                                              * true_obj / tf.reduce_sum(true_obj)

    wh_loss = self.lambda_coord * tf.reduce_sum(tf.square(true_wh_rel-pred_wh_rel), axis=-1) \
                                              * true_obj / tf.reduce_sum(true_obj)
    
    obj_loss = tf.reduce_sum(K.binary_crossentropy(true_obj, pred_obj) * true_obj)
    noobj_loss = self.lambda_noobj * tf.reduce_sum(K.binary_crossentropy(true_obj, pred_obj) * (1- true_obj) * ignore_mask)

    class_loss = tf.reduce_sum(K.binary_crossentropy(true_class, pred_class) * true_obj)

    return xy_loss + wh_loss + obj_loss + noobj_loss + class_loss, xy_loss, wh_loss, obj_loss, noobj_loss, class_loss

  def calc_ignore_mask(self, y_pred, y_true):
    output_shape = y_pred.shape
    output_shape[-1] = 1

    pred_bbox = y_pred[..., :4]
    pred_bbox[..., :2] = tf.sigmoid(pred_bbox[..., :2])
    pred_bbox = rel_to_abs_box(pred_bbox, self.grid, self.anchors) # (batch, grid, grid, 3, 4)
    pred_bbox = xywh_to_x1y1x2y2(pred_bbox) # convert to x1y1x2y2
    pred_bbox = tf.reshape(pred_bbox, [pred_bbox.shape[0], -1, 4]) # (Batch, grid*grid*3, 4)

    true_bbox = y_true[..., :4]
    true_bbox = rel_to_abs_box(true_bbox, self.grid, self.anchors) # (batch, grid, grid, 3, 4)
    true_bbox = xywh_to_x1y1x2y2(true_bbox) # convert to x1y1x2y2
    true_bbox = tf.reshape(true_bbox, [true_bbox.shape[0], -1, 4]) # (Batch, grid*grid*3, 4)
    true_bbox = tf.sort(true_bbox, -2, direction='DESCENDING')[:, :100, :]

    ious = self.calc_iou(pred_bbox, true_bbox)
    best_iou = tf.reduce_max(ious, axis=-1)
    best_iou = tf.reshape(best_iou, output_shape)
    
    ignore_mask = tf.cast(best_iou < self.ignore_thresh, dtype=tf.float32)
    return ignore_mask
  
  def calc_iou(self, box_a, box_b):
    """
    calculate iou between box_a and multiple box_b in a broadcast way.
    Used this implementation as reference: 
    https://github.com/dmlc/gluon-cv/blob/c3dd20d4b1c1ef8b7d381ad2a7d04a68c5fa1221/gluoncv/nn/bbox.py#L206
    inputs:
    box_a: a tensor full of boxes, eg. (B, N, 4), box is in x1y1x2y2
    box_b: another tensor full of boxes, eg. (B, M, 4)
    """

    # (B, N, 1, 4)
    box_a = tf.expand_dims(box_a, -2)
    # (B, 1, M, 4)
    box_b = tf.expand_dims(box_b, -3)
    # (B, N, M, 4)
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_a), tf.shape(box_b))

    # (B, N, M, 4)
    # (B, N, M, 4)
    box_a = tf.broadcast_to(box_a, new_shape)
    box_b = tf.broadcast_to(box_b, new_shape)

    # (B, N, M, 1)
    al, at, ar, ab = tf.split(box_a, 4, -1)
    bl, bt, br, bb = tf.split(box_b, 4, -1)

    # (B, N, M, 1)
    left = tf.math.maximum(al, bl)
    right = tf.math.minimum(ar, br)
    top = tf.math.maximum(at, bt)
    bot = tf.math.minimum(ab, bb)

    # (B, N, M, 1)
    iw = tf.clip_by_value(right - left, 0, 1)
    ih = tf.clip_by_value(bot - top, 0, 1)
    i = iw * ih

    # (B, N, M, 1)
    area_a = (ar - al) * (ab - at)
    area_b = (br - bl) * (bb - bt)
    union = area_a + area_b - i

    # (B, N, M)
    iou = tf.squeeze(i / (union + 1e-7), axis=-1)

    return iou









