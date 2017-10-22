import caffe

import numpy as np
from PIL import Image as Im

import random


class YOLO2CaffeLayer(caffe.Layer):


    def setup(self, bottom, top):
	self.kkk = 0
        # config
        params = eval(self.param_str)
        self.dataset = params['test']
	self.lexicon = params['lexicon']
	self.batch = params['batch_size']

        self.data = open(self.dataset, 'r').readlines()#.splitlines()
	self.data = [o.rstrip("\n").split(",") for o in self.data]

        self.idx = 0

 
    def reshape(self, bottom, top):
        # load image + label image pair
	top[0].reshape(self.batch, 3, 416, 416)
        top[1].reshape(self.batch, 28)


    def loadImage(self):
	filename =  self.data[self.idx][0]

        #print filename

        new_im = np.ones((416,416,3))*0.5
        im = Im.open(filename)
        w = im.size[0]
        h = im.size[1]

        if im.size[0] > im.size[1]: # w>h
                new_w = 416
                new_h = int(416*h/w)
                nim = im.resize((new_w, new_h), Im.BILINEAR)
		
                for w in range(416):
                        for h in range(new_h):
				
                                new_im[w,208-new_h/2+h-1,...] =  np.array(nim.getpixel((w,h)))/255.0#np.array([nim.getpixel((w,h))/255.0,nim.getpixel((w,h))/255.0,nim.getpixel((w,h))/255.0])

        else:
                new_h = 416
                new_w = int(416*w/h)
                nim = im.resize((new_w, new_h), Im.BILINEAR)


                for h in range(416):
                        for w in range(new_w):
                                new_im[208-new_w/2+w-1,h,...] = np.array(nim.getpixel((w,h)))/255.0

        new_im = new_im[:,:,::-1]
        #print new_im.shape
	
        new_im = new_im.transpose((2, 1, 0))
	
	return new_im

    def loadLabel(self):
	label = np.zeros(28)
	label[int(self.data[self.idx][1])] = 1.0
	return label

    def forward(self, bottom, top):
        # assign output
		
	for b in range(self.batch):
		top[0].data[b,...] = self.loadImage()
		top[1].data[b,...] = self.loadLabel()
		#print "load label, ", top[1].data[b,...]


        # pick next input
	        self.idx += 1
        	if self.idx == len(self.data):
	        	self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    
