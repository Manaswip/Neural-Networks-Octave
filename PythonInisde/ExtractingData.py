import os,struct
from array import array as pyarray
import numpy as np

def ExtractingData():
	img_data = os.path.join('.','train-images-idx3-ubyte')
	lbl_data = os.path.join('.','train-labels-idx1-ubyte')

	file_img = open(img_data,'rb')
	magic_nr, size,rows,cols = struct.unpack(">IIII",file_img.read(16))
	img = pyarray("b",file_img.read())
	file_img.close()


	file_lbl = open(lbl_data,'rb')
	magic_nr,size = struct.unpack(">II",file_lbl.read(8))
	lbl = pyarray("B",file_lbl.read())
	file_lbl.close()

	digits = np.arange(10)

	ind = [ k for k in range(size) if lbl[k] in digits ]
	N = len(ind)

	images = np.zeros((N,rows*cols),dtype=np.uint8)
	labels = np.zeros((N,1),dtype=np.uint8)

	for i in range(len(ind)):
		images[i] = np.array(img[ind[i]*rows*cols : (ind[i]+1)*rows*cols])
		labels[i] = lbl[ind[i]]

	return images,labels



