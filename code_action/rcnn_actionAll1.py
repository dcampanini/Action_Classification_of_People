#!/usr/bin/env python
# coding: utf-8

import caffe
import numpy as np
from StringIO import StringIO # para guardar arreglos en txt
import cPickle as pickl # para guardar
import matplotlib.pyplot as plt
import xml.etree.ElementTree # para abrir xml

# Make sure that caffe is on the python path:
caffe_root='/home/diego/caffe-master'
import sys
sys.path.insert(0, caffe_root + 'python')
print(caffe_root + 'python')

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os
caffe.set_mode_cpu()

# Prueba en la red de detección de acciones Net 2
# .prototxt es la arquitectura de la red
# pascal_finetune_HumanNet2_iter_10000 son los pesos de la red
net = caffe.Net('pascal_finetune_fc8.prototxt',
                'pascal_finetune_HumanNet2_iter_10000',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 50
# .reshape(bach,n° de caneles (=3 si es RGB),height, width)
net.blobs['data'].reshape(1,3,227,227)

# leer archivo txt
from PIL import Image # para usar libreria de imágenes
import numpy as num
arch=open('test2.txt','r')# leer txt conjunto de test
c=0
linea=arch.readline()# leer una linea del archivo test2.txt
lista=linea.split()# pasar la línea a una lista

while linea!="":
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('/home/diego/caffe-master/examples/VOC2012_test2/'+lista[0]+'_'+lista[1]+'.jpg'))# cargar imagen para ser analizada
    out = net.forward()# salida desde la ultima capa de la red
    c=c+1

    print("Predicted class is #{}.".format(out['prob'].argmax()))# prob es el nombre de la ultima capa del modelo de red cnn
                                                                 # argmax es para obtener la salida con maxima probabilidad
    print("Real class is #{}.".format(lista[2]))
    # leer la linea siguiente
    linea=arch.readline();
    lista=linea.split();

    # Para el programa para no analizar todas las imagenes
    if c>=10:
        break
