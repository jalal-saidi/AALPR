import sys
import cv2
import numpy as np
import traceback
import multiprocessing

from multiprocessing import Queue
from VideoSource import VideoGet
from aalpr_task import AALPR_Task
#dn.set_gpu(0)

from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder

class Vehicle_Detector(multiprocessing.Process):
	def __init__(self, vehicle_threshold,vehicle_weights,vehicle_netcfg,vehicle_dataset, work_queue, result_queue, src=0  ):
		multiprocessing.Process.__init__(self)
		self.vehicle_weights = vehicle_weights
		self.vehicle_netcfg = vehicle_netcfg
		self.vehicle_threshold = vehicle_threshold
		self.vehicle_dataset = vehicle_dataset
		self.result_queue = result_queue
		self.data_source = VideoGet(src).start()
		self.num_frames = 0
		self.vehicle_net = None
		self.vehicle_meta = None
		self.total_cars = 0
		#print 'loaded the network'
		self.vehicle_times = open('vehicle_times.txt', 'w')
	def run(self):
		from darknet.python.darknet import detect
		import darknet.python.darknet as dn
		self.vehicle_net = dn.load_net(self.vehicle_netcfg, self.vehicle_weights, 0)
		self.vehicle_meta = dn.load_meta(self.vehicle_dataset)
		try:
			while True:
				#print 'reading from queue'
				if self.data_source.more():
					frame = self.data_source.get()
				else:
					continue
				if frame is None:
					print 'frame was none, total_cars', self.total_cars
					self.result_queue.put(None)
					self.vehicle_times.close()
					break
				self.num_frames += 1
				#frame = task
				import datetime
				start = datetime.datetime.now()
				#print 'going for detect shape is ', frame.shape[1::-1], " ", self.num_frames
				R, _ = detect(self.vehicle_net, self.vehicle_meta, frame, thresh=self.vehicle_threshold)
				end = datetime.datetime.now()
				elapsed = end - start
				self.vehicle_times.write('vehicle_detection_time %d\n'%(elapsed.microseconds//1000))
				print 'vehicle_detection_time %d, num_frames %d\n'%((elapsed.microseconds//1000), self.num_frames)
				self.vehicle_times.flush()
				# using only cars and buses to the queue
				R = [r for r in R if r[0] in ['car', 'bus']]
				print '\t\t%d cars found, num_frames %d\n' % (len(R), self.num_frames)
				self.total_cars += len(R)
				task2 = AALPR_Task(frame)
				if len(R):
					WH = np.array(frame.shape[1::-1], dtype=float)
					Lcars = []
					for i, r in enumerate(R):
						#print r
						cx, cy, w, h = (np.array(r[2]) / np.concatenate((WH, WH))).tolist()
						tl = np.array([cx - w / 2., cy - h / 2.])
						br = np.array([cx + w / 2., cy + h / 2.])
						label = Label(0, tl, br)
						#tl_cord = np.floor(label.tl() * WH).astype(int)
						#br_cord = np.ceil(label.br() * WH).astype(int)
						#tl_cord = np.maximum(tl_cord,0).astype(int)
						#br_cord = np.minimum(br_cord,WH).astype(int)
						#cv2.rectangle(frame, (tl_cord[0],tl_cord[1]), (br_cord[0], br_cord[1]),(0,255,0), 4 )

						#Icar = crop_region(frame, label)
						Lcars.append(label)
						task2.add_vehicle(label)

				self.result_queue.put(task2)
		except:
			traceback.print_exc()
						#cv2.imwrite('%d_%dcar.png' % (frame_num,i), frame)
						#cv2.imshow('image', frame)
						#cv2.imshow('cropped', frame)



