import sys, os

import cv2
import traceback
import multiprocessing
import numpy as np

class Char_Classifier(multiprocessing.Process):
	def __init__(self, char_threshold,char_net_path, char_class_names_path,  work_queue, result_queue  ):
		multiprocessing.Process.__init__(self)

		self.char_threshold = char_threshold
		self.char_net_path = char_net_path
		self.work_queue = work_queue
		self.result_queue = result_queue
		self.char_class_names_list = []
		with open(char_class_names_path, 'r') as class_file:
			self.char_class_names_list = class_file.readlines()
		self.char_class_names_dict = {idx: name.strip() for idx, name in enumerate(self.char_class_names_list)}
		print self.char_class_names_dict
		self.num_processed = 0
		self.num_frames = 0
		self.char_times = open('char_times.txt', 'w')
		self.outVid = None
		#self.outVid = cv2.VideoWriter('outpy.mp4', 0x00000021, 15.0, (1920,1080))



	def run(self):
		from src.utils import im2single
		from src.keras_utils import load_model
		self.outVid = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1920, 1080))

		import datetime
		from src.label import Shape, writeShapes
		self.char_net = load_model(self.char_net_path)

		try:
			while True:
				print 'char classifier is running'

				task = self.work_queue.get()
				if task is None:
					print 'task was none'
					self.result_queue.put(None)
					self.char_times.close()
					self.outVid.release()
					break
				print "char classifier enumerating in all char labels ", self.num_frames
				fr = task.frame
				self.num_frames += 1
				for i , lp_char_label in enumerate(task.get_all_lp_char_labels()):
					self.num_processed += 1
					lp = task.get_lp(i)
					if lp is not None:
						offset = lp.shape[1] * i
						fr[0:lp.shape[0], offset:offset + lp.shape[1]] = lp
					print "got lp char images and classifying them ", i
					(lp_char_imgs,lp_char_imgs2)  = task.get_lp_char_imgs(i)
					if len(lp_char_imgs):

						start = datetime.datetime.now()
						lp_char_res = []
						print "predicting with model len is ", len(lp_char_imgs)
						for i,j in zip(lp_char_imgs, lp_char_imgs2):
							if i is not None:
								res =self.char_class_names_dict[ np.argmax(self.char_net.predict(i))]
								lp_char_res.append(res)
								Ilp = cv2.cvtColor(j, cv2.COLOR_BGR2GRAY)
								Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
							else:
								print 'i was None'
						image_txt = " ".join(lp_char_res)

						print 'the plate is ', image_txt
						cv2.putText(fr, image_txt, (offset,lp.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
						end = datetime.datetime.now()
						elapsed = end - start
						self.char_times.write("lp_detect_time %d\n"%(elapsed.microseconds//1000))
						self.char_times.flush()

						#print 'detect_lp returned'
						if len(lp_char_res):
							pass

						else:
							task.add_lps(None)
				print 'char classifier putting the task into queue', self.num_frames
				#cv2.imwrite('%d.png' % (self.num_processed), fr)
				self.outVid.write(fr)

			#self.result_queue.put(task)
			self.outVid.release()
		except:
			traceback.print_exc()
