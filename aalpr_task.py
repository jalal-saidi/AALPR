# Single Task in Afghan Automatic License Plate Recognition (AALPR) System
# It includes captured frames, detected vehicles, license plates, and char lables
from src.utils 	import crop_region
import numpy as np
import cv2
class AALPR_Task(object):
	def __init__(self, frame):
		self.frame = frame
		self.vehicles = []
		self.vehicle_labels = []
		self.lps = []
		self.lp_char_labels = []
		self.popped_vehicles = 0
		self.num_processed = 0

	# vehicles detected by Vehicle Detection
	def add_vehicle(self,label):
		self.vehicle_labels.append(label)

	def has_vehicles(self):
		return self.popped_vehicles < len(self.vehicle_labels)

	# vehicles detected by Vehicle Detection
	def get_frame(self):
		return self.frame

	
	def get_vehicles(self):
		return self.vehicle_labels

	def get_vehicle_img(self, i):
		Icar = crop_region(self.frame, self.vehicle_labels[i])
		return Icar.astype('uint8')

	def get_lp(self, i):
		return self.lps[i]

	def get_all_lps(self):
		return self.lps

	def add_lps(self, lp):
		self.lps.append(lp)

	def add_lp_char(self, lp_char_label):
		self.lp_char_labels.append(lp_char_label)

	def get_lp_char_imgs(self, i):
		lp = self.lps[i]
		self.num_processed +=1
		char_imgs = []
		char_imgs2  = []
		if self.lp_char_labels[i] is not None and lp is not None:
			R, (w, h) = self.lp_char_labels[i]
			for r in R:
				if r is not None:
					center = np.array(r[2][:2])
					wh2 = (np.array(r[2][2:])) * .5
					tl = tuple((center - wh2).astype(int))
					br = tuple((center + wh2).astype(int))
					char_img = lp[tl[1]:br[1], tl[0]:br[0]]
					old_size = char_img.shape[:2]  # old_size is in (height, width) format

					if old_size[0] == 0 or old_size[1] == 0:
						char_imgs.append(None)
						char_imgs2.append(None)
						continue

					desired_size = 30
					ratio = float(desired_size) / max(old_size)
					new_size = tuple([int(x * ratio) for x in old_size])

					# new_size should be in (`idth, height) format
					print 'old ', char_img.shape, ' ', 'new ', new_size

					char_img2 = cv2.resize(char_img, (new_size[1], new_size[0]))

					delta_w = desired_size - new_size[1]
					delta_h = desired_size - new_size[0]
					top, bottom = delta_h // 2, delta_h - (delta_h // 2)
					left, right = delta_w // 2, delta_w - (delta_w // 2)

					color = [0, 0, 0]
					new_im = cv2.copyMakeBorder(char_img2, top, bottom, left, right, cv2.BORDER_CONSTANT,
												value=color).astype('float32')/255.

					char_imgs2.append(new_im)
					new_im = new_im.reshape((1, new_im.shape[0], new_im.shape[1], new_im.shape[2]))

					char_imgs.append(new_im)
				else:
					char_imgs.append(None)
					char_imgs2.append(None)

		return (char_imgs, char_imgs2)

	def get_all_lp_char_labels(self):
		return self.lp_char_labels



