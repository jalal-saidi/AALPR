import os
import cv2
import numpy as np

class_names = '/Volumes/hd2/projects/nimas_plates/classes.txt'
input_dir = '/Volumes/hd2/projects/labeled_chars'
output_dir = '/Volumes/hd2/projects/labeled_char_dark/train'
output_dir_test = '/Volumes/hd2/projects/labeled_char_dark/test'
class_id_name_dict = {}
file_list = os.listdir(input_dir)
with open(class_names, 'r') as class_list:
	for id, label_name in enumerate(class_list.readlines()):
		print id, label_name
		class_id_name_dict[id] = label_name.strip()
print class_id_name_dict
img_counter = 0
test_counter = 0
for file_name in file_list:
	if file_name.endswith(".png"):
		img = cv2.imread(input_dir+"/"+file_name)
		label_id = int(file_name.split('_')[2].split(".")[0])
		label_name = class_id_name_dict[label_id]
		if img_counter < 4500:
			new_file_name = output_dir+"/"+"%d_%s.png"%(img_counter, label_name)
		else:
			new_file_name = output_dir_test+"/"+"%d_%s.png"%(test_counter, label_name)
			test_counter+=1
		cv2.imwrite(new_file_name,img)
		img_counter += 1
