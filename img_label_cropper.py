import cv2
import os
import numpy as np

path = '/Volumes/hd2/projects/nimas_plates'
out_dir = '/Volumes/hd2/projects/labeled_chars'
def crop_and_save_labels(image_name, label_name, out_dir):
	#handle the image names
	full_image_name = path+"/"+image_name
	img_no_ext = image_name.split(".")[0]
	full_image_label_name= path+"/"+label_name

	#read the image from disk
	img = cv2.imread(full_image_name)
	WH = np.array([240,80], dtype=float)


	#open the label file
	with open(full_image_label_name,'r') as labels:
		print 'working on '+ full_image_label_name
		for label in labels.readlines():
			print label
			label_parts = map(float, label.split()[1:])
			center = label_parts[:2] * WH
			wh = label_parts[2:] * WH *.5
			tl = tuple((center-wh).astype(int))
			br = tuple((center+wh).astype(int))
			label_txt = label.split()[0].strip()
			print tl, br, label_txt
			crp = img[tl[1]:br[1], tl[0]:br[0]]
			output_file_name = out_dir+"/"+img_no_ext+"_"+label_txt+".png"
			cv2.imwrite(output_file_name,crp)
			print output_file_name
			#img1 = cv2.rectangle(img, tl, br, (0,0,0),1)
			#output_file =
			print label_parts




file_names = os.listdir(path)
file_names_set = set(file_names)
img_label_file_dict = {}



for file_name in file_names:
	if file_name.endswith(".png"):
		file_label_name = file_name.split(".")[0] + ".txt"
		if file_label_name in file_names_set:
			img_label_file_dict[file_name]= file_label_name

for image_name, label_name in img_label_file_dict.iteritems():
	crop_and_save_labels(image_name, label_name, out_dir)
