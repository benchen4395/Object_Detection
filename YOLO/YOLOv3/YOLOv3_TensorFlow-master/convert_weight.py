# coding: utf-8
"""
Created on Tue Oct 23 17:24:09 2018
Refer: https://blog.csdn.net/leviopku/article/details/82660381
@author: benchen
"""

# for more details about the yolo darknet weights file, refer to
# https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe

from __future__ import division, print_function

import os
import sys
import tensorflow as tf
import numpy as np

from model import yolov3
from utils.misc_utils import parse_anchors, load_weights

num_class = 80
img_size = 416
weight_path = '../yolov3.weights'
save_path = './data/tensorflow_weights/yolov3.ckpt'
anchors = parse_anchors('./data/yolo_anchors.txt') # 将anchors编程（-1,2）的数据

model = yolov3(80, anchors)
with tf.Session() as sess:
    inputs = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
    # 这里描述一下，为什么要加这一步，见B1_P108.  这一步很迷，不加的时候是否可以？
    # 生成一个上下文管理器，在这个管理器中，forward中的值将直接获取已经生成的变量
    with tf.variable_scope('yolov3'):
        # feature_map是一个len=3的数组，每一个元素都对应着一个scale上的prediction
        feature_map = model.forward(inputs)

    saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))

    load_ops = load_weights(tf.global_variables(scope='yolov3'), weight_path)
    sess.run(load_ops)
    saver.save(sess, save_path=save_path)
    print('TensorFlow model checkpoint has been saved to {}'.format(save_path))



