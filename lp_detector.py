import sys, os

import cv2
import traceback
import multiprocessing


def adjust_pts(pts, lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))

# A multiprocess object for License Plate Detection


class LP_Detector(multiprocessing.Process):
	def __init__(self, lp_threshold,wpod_net_path, work_queue, result_queue  ):
		multiprocessing.Process.__init__(self)

		self.lp_threshold = lp_threshold
		self.wpod_net_path = wpod_net_path
		self.work_queue = work_queue
		self.result_queue = result_queue
		self.num_processed = 0
		self.lptimes = open('lp_times.txt', 'w')
		self.num_frames = 0

	def run(self):
		from src.utils import im2single
		from src.keras_utils import load_model, detect_lp
		from src.label import Shape, writeShapes
		self.wpod_net = load_model(self.wpod_net_path)

		try:
			#print 'Searching for license plates using WPOD-NET'
			while True:
				#print 'lp detector is running'

				task = self.work_queue.get()
				if task is None:
					print 'task was none'
					self.result_queue.put(None)
					self.lptimes.close()
					break
				self.num_frames +=1
				for i , vehicle_labels in enumerate(task.get_vehicles()):
					self.num_processed += 1
					Ivehicle = task.get_vehicle_img(i)
					if Ivehicle is not None:
						ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
						side = int(ratio * 288.)
						bound_dim = min(side + (side % (2 ** 4)), 608)
						import datetime
						start = datetime.datetime.now()

						#print "\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio)
						Llp, LlpImgs, _ = detect_lp(self.wpod_net, im2single(Ivehicle),
												bound_dim, 2 ** 4, (240, 80), self.lp_threshold)
						end = datetime.datetime.now()
						elapsed = end - start
						self.lptimes.write("lp_detect_time %d\n"%(elapsed.microseconds//1000))
						self.lptimes.flush()

						#print 'detect_lp returned'
						if len(LlpImgs):
							#print 'writing image to disk'
							Ilp = LlpImgs[0]
							Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
							Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
							task.add_lps(Ilp*255.)
							#cv2.imshow('new', Ilp)
							#cv2.waitKey(0)
							#s = Shape(Llp[0].pts)

							#cv2.imwrite('%d_lp.png' % (self.num_processed ), Ilp * 255.)
							#writeShapes('%d_lp.txt' % (self.num_processed), [s])
							#put into result queue
						else:
							task.add_lps(None)
				print 'lp detector putting the task into queue ', self.num_frames
				self.result_queue.put(task)
		except:
			traceback.print_exc()