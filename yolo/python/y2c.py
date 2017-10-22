# load yolo v2 weight file(binary)
import numpy as np
import sys, getopt
import sys

import math
import caffe
from caffe import layers as L, params as P # Shortcuts to define the net prototxt.
import operator
import os
import re
import PIL.Image as Im
import datetime


def create_testing_file(test_data_path, lexicon_path, batch_size, input_size, num_output): #yolo_prototxt, num_output):
	
	prototxt_content = open("yolo/y2c_template.prototxt", "r").readlines()

	with open("yolo/y2c_output_"+str(num_output)+".prototxt", "w") as o:
		for c in prototxt_content:
			if "change here" in c:
				o.write("    param_str: \"{\\\'test\\\': \\\'"+test_data_path+"\\\', \\\'batch_size\\\': "+str(batch_size)+ ",\\\'lexicon\\\': \\\'"+lexicon_path+"\\\'}\"\n") 
			else:
				o.write(c)
	print "------------------------------------------"
	print "Prototxt Created! Path:","yolo/y2c_output_"+str(num_output)+".prototxt"
	
	solver_content = open("yolo/solver_template.prototxt", "r").readlines()
	with open("yolo/solver.prototxt", "w") as o:
		for c in solver_content:
			if "test_net" in c:
				o.write('test_net: "yolo/y2c_output_'+str(num_output)+'.prototxt"\n')	
			elif "train_net" in c:
				o.write('train_net: "yolo/y2c_output_'+str(num_output)+'.prototxt"\n')	
			else:
				o.write(c)
	
	print "------------------------------------------"
	print "Solver Created! Path:","yolo/solver.prototxt"
	print "------------------------------------------"

def softmax(arr):
	out = [math.exp(a) for a in arr]
	return (np.array(out)/sum(out))

def logistic(arr):
	return np.array([1./(1+math.exp(-x)) for x in arr])
	
def check_accuracy(net, test_length, batch, num_output):
	
	hit = 0.0
	count = 0.0
	fg = 0
	for g in range(test_length/batch+1):

		net.forward()
		ests = net.blobs['conv_reg'].data
		est_output = np.zeros((num_output*5,13,13))

		for f in range(batch):
			for i in range(13):
				for j in range(13):
					for t in range(5):
						part_s = (num_output+5)*t
						for k in range(5, num_output+5):
							est_output[num_output*t+k-5][i][j] = ests[0][part_s][i][j]*ests[0][part_s+k][i][j]					
					est_output[...,i,j] = softmax(logistic(est_output[...,i,j]))

			count += 1
			if count > test_length:
				fg = 1
				break

			prediction = (np.argmax(est_output)/169)%num_output
			hit += (net.blobs['label'].data[0][prediction] == 1.0)

		if fg == 1:
			break			
		
		dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		print dt, g, hit/((g+1)*batch)
		
	return hit/(test_length)
	
def transform(yolo_weights, yolo_prototxt, caffemodel_path):
	yolo_wei = np.fromfile(yolo_weights, dtype=np.float32)[4:]
	print "YOLO total weights : ",len(yolo_wei)
	print "------------------------------------------"

	net = caffe.Net(yolo_prototxt, caffe.TEST)  
	params = net.params.keys()
	print "Layers: ", params
	print "------------------------------------------"


	param_end = 0
	# yolo save param order: bias -> scale -> mean -> variance -> weights
	for i in range(1,22):

	    wei_shape = net.params['conv'+str(i)][0].data.shape	
	    bias_shape = net.params['scale_conv'+str(i)][1].data.shape
            mean_shape = net.params['bn'+str(i)][0].data.shape
            var_shape = net.params['bn'+str(i)][1].data.shape
            scale_shape = net.params['scale_conv'+str(i)][0].data.shape
	    

            net.params['scale_conv'+str(i)][1].data[...] = yolo_wei[param_end:param_end+np.prod(bias_shape)].reshape(bias_shape)
	    param_end += np.prod(bias_shape)

            net.params['scale_conv'+str(i)][0].data[...] = yolo_wei[param_end:param_end+np.prod(scale_shape)].reshape(scale_shape)
	    param_end += np.prod(scale_shape)
	    
            net.params['bn'+str(i)][0].data[...] = yolo_wei[param_end:param_end+np.prod(mean_shape)].reshape(mean_shape)
            param_end += np.prod(mean_shape)

            net.params['bn'+str(i)][1].data[...] = yolo_wei[param_end:param_end+np.prod(var_shape)].reshape(var_shape)
            param_end += np.prod(var_shape)

            net.params['bn'+str(i)][2].data[0] = 1.0  

	    net.params['conv'+str(i)][0].data[...] = yolo_wei[param_end:param_end+np.prod(wei_shape)].reshape(net.params['conv'+str(i)][0].data.shape)#.transpose((0,1,3,2))
            param_end += np.prod(wei_shape)

	conv_reg_wei_shape = net.params['conv_reg'][0].data.shape
	conv_reg_bias_shape = net.params['conv_reg'][1].data.shape

	net.params['conv_reg'][1].data[...] = yolo_wei[param_end:param_end+np.prod(conv_reg_bias_shape)].reshape(conv_reg_bias_shape)
	param_end += np.prod(conv_reg_bias_shape)

	net.params['conv_reg'][0].data[...] = yolo_wei[param_end:param_end+np.prod(conv_reg_wei_shape)].reshape(conv_reg_wei_shape)
	param_end += np.prod(conv_reg_wei_shape)


	if os.path.isfile(caffemodel_path):
		os.remove(caffemodel_path)	
	print "Model Matching:",
	if param_end == len(yolo_wei):
	    print "Success!!"
            net.save(caffemodel_path)
	else:
	    print "Error!!"


