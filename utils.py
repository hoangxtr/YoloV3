import tensorflow as tf
import glob
import os
import xml.etree.ElementTree as ET
import pandas as pd
import io
from PIL import Image
from collections import namedtuple
from config import *

map_class = {'Type 1': 1, 'Type 2': 2}
YOLO_ANCHORS = [[[10,  13], [16,   30], [33,   23]],
                [[30,  61], [62,   45], [59,  119]],
                [[116, 90], [156, 198], [373, 326]]]
ANCHORS = (np.array(YOLO_ANCHORS)/416)

def rel_to_abs_box(bbox, grid, anchors):
  '''
    bbox: shape(batch, grid, grid, 3, 4)
  '''
  t_xy = bbox[..., :2] # (batch, grid, grid, 3, 2)
  t_wh = bbox[..., 2:4] # (batch, grid, grid, 3, 2)

  c_xy = tf.meshgrid(tf.range(grid), tf.range(grid)) # (grid, grid)
  c_xy = tf.stack(c_xy, axis=-1) # (grid, grid, 2)
  c_xy = tf.cast(tf.expand_dims(c_xy, -2), dtype=tf.float32) # (grid, grid, 1, 2)
  b_xy = (t_xy + c_xy) / tf.cast(grid, tf.float32) # (batch, grid, grid, 3, 2) (c_xy can auto duplicate for match new size)

  b_wh = tf.exp(t_wh) * anchors
  return tf.concat((b_xy, b_wh), axis=-1)

def x1y1x2y2_to_xywh(bbox):
  x1y1 = bbox[..., :2]
  x2y2 = bbox[..., 2:4]
  xy = (x1y1 + x2y2) / 2.0
  wh = x2y2-x1y1
  return tf.concat([xy,wh], axis=-1)

def get_anchor_of_grid(grid):
  return ANCHORS[int(np.log2(grid/13))] 

def calc_gious(box_wh, anchor_wh):
  '''
  box_wh: (batch, num_box, 3, 2)
  anchor_wh: (3, 2)
  '''

  intersections = tf.minimum(box_wh[..., 0], anchor_wh[..., 0]) * tf.minimum(box_wh[..., 1], anchor_wh[..., 1])
  box_area = box_wh[..., 0] * box_wh[..., 1]
  anchor_area = anchor_wh[..., 0] * anchor_wh[..., 1]
  gious = tf.cast(intersections / anchor_area, tf.float32)
  return gious 

def xywh_to_x1y1x2y2(bbox):
  xy = bbox[..., :2]
  wh = bbox[..., 2:4]
  x1y1 = xy - wh / 2
  x2y2 = x1y1 + wh
  return tf.concat([x1y1, x2y2], axis=-1) 

def xml_to_csv(paths):
  content = []
  for path in paths:
    tree = ET.parse(path)
    root = tree.getroot()
    
    for obj in root.findall('object'):
      value = [
               root.find('filename').text,
               int(root.find('size')[0].text),
               int(root.find('size')[1].text),
               obj[0].text,
               int(obj[4][0].text),
               int(obj[4][1].text),
               int(obj[4][2].text),
               int(obj[4][3].text),
      ]
      content.append(value)
  column = ['filename', 'width', 'height', 'type', 'x1', 'y1', 'x2', 'y2']
  return pd.DataFrame(content, columns=column)


def split(df, group):
  data = namedtuple('data', ['filename', 'object'])
  gb = df.groupby(group)
  return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tfrecord(group, path):
  with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = Image.open(encoded_jpg_io)
  width, height = image.size

  filename = group.filename.encode('utf8')
  image_format = b'jpg'
  xmins = []
  xmaxs = []
  ymins = []
  ymaxs = []
  classes_text = []
  classes = []

  for index, row in group.object.iterrows():
    xmins.append(row['x1'] / width)
    xmaxs.append(row['x2'] / width)
    ymins.append(row['y1'] / height)
    ymaxs.append(row['y2'] / height)
    classes_text.append(row['type'].encode('utf8'))
    classes.append(map_class[row['type']])
  
  tf_example = tf.train.Example(features=tf.train.Features(
      feature={
          'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
          'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
          'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
          'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
          'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
          'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
          'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
          'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
          'image/object/bbox/class': tf.train.Feature(int64_list=tf.train.Int64List(value=classes))
      }
  ))
  return tf_example














