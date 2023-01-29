import sys, os

import cv2
import traceback
import multiprocessing
import numpy as np

def adjust_pts(pts, lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))

def sort_boxes(a):
	return a[2][0]


# A multiprocess object for License Plate Segmentation

class LP_Segmentator(multiprocessing.Process):
	def __init__(self, seg_thershold, seg_weights, seg_netcfg,seg_dataset, work_queue, result_queue):
		multiprocessing.Process.__init__(self)
		import darknet.python.darknet as dn
		self.seg_weights = seg_weights
		self.seg_netcfg = seg_netcfg
		self.seg_threshold = seg_thershold
		self.seg_dataset = seg_dataset
		self.seg_net = None
		self.seg_meta = None
		self.work_queue = work_queue
		self.result_queue = result_queue
		self.num_chars = 0
		self.num_frames = 0
		self.lp_char_times = open('lp_char_times.txt','w')

	def run(self):
		from darknet.python.darknet import detect
		import darknet.python.darknet as dn

		self.seg_net = dn.load_net(self.seg_netcfg, self.seg_weights, 0)
		self.seg_meta = dn.load_meta(self.seg_dataset)

		try:
			while True:
				#print 'reading lps from the queue for segmentation'
				task = self.work_queue.get()
				if task is None:
					self.result_queue.put(None)
					self.lp_char_times.close()
					print 'task was none'
					break
				num_plates = 0
				self.num_frames +=1
				for i, lp in enumerate(task.get_all_lps()):
					if lp is not None:

						#print 'going to print lps'

						#cv2.imshow('img1', lp)
						#cv2.waitKey(0)
						import datetime
						start = datetime.datetime.now()
						#print 'lp_seg for detection'
						res = detect(self.seg_net, self.seg_meta, lp, self.seg_threshold, nms=0.5)

						end = datetime.datetime.now()
						elapsed = end - start
						self.lp_char_times.write("lp_char_time %d\n"%(elapsed.microseconds//1000))
						self.lp_char_times.flush()
						#print 'detected in lp_segmentator'

						R, (w, h) = res
						res = (sorted(R, key=sort_boxes), (w, h))
						task.add_lp_char(res)
						#print "unsorted R" , R
						#print "sorted R ", sorted(R, key=sort_boxes)
						WH = np.array([w, h], dtype=float)
						#print 'num detected chars' , len(R)
						img1 = lp
						for r in R:
							#print 'going to write chars to disk '
							#print r[2]
							center = np.array(r[2][:2])
							wh2 = (np.array(r[2][2:])) * .5
							tl = tuple((center - wh2).astype(int))
							br = tuple((center + wh2).astype(int))
							#print tl
							# tl[0] = int(tl[0])
							# tl[1] = int(tl[1])
							# br[0] = int(br[0])
							# br[1] = int(br[1])

							#img1 = cv2.rectangle(img1, tl, br, (0, 0, 0), 1)
						#offset = img1.shape[1] * num_plates
						#fr[0:img1.shape[0], offset:offset+img1.shape[1]] = img1
						num_plates += 1
				self.num_chars += 1
				#self.outVid.write(fr)
				#cv2.imwrite("output/img-%04d.jpg"%self.num_chars, fr)
							#cv2.imshow('img1', img1)
							#cv2.waitKey(0)
				print 'lp segmentor putting the chars into the result queue ', self.num_frames
				self.result_queue.put(task)


		except:
			traceback.print_exc()

	def test(self):
		from darknet.python.darknet import detect
		import cv2
		img1 = cv2.imread('1_lp.png')
		R, (w, h) =  detect(self.seg_net, self.seg_meta,img1 , self.seg_threshold, nms=0.45)
		return (R, (w, h), img1)


if __name__ == '__main__':
	lps = LP_Segmentator(0.4, 'data/ocr2/ocr-net_final.weights', 'data/ocr2/ocr-net.cfg','data/ocr2/ocr-net.data', None, None)
	import cv2
	R, (w, h), img1  = lps.test()
	WH = np.array([w,h],dtype=float)


	#print R
	for r in R:
		print r[2]
		center = np.array(r[2][:2])
		wh2 = (np.array(r[2][2:]))*.5
		tl = tuple((center - wh2).astype(int))
		br = tuple((center + wh2).astype(int))
		print tl
		# tl[0] = int(tl[0])
		# tl[1] = int(tl[1])
		# br[0] = int(br[0])
		# br[1] = int(br[1])

		img1 = cv2.rectangle(img1, tl, br, (0,0,0), 1)
		#crp = img1[tl[1]:br[1], tl[0]:br[0]]
		#cv2.imshow('crop',crp)
		#cv2.waitKey(0)
	cv2.imshow('img1', img1)
	cv2.waitKey(0)
	print w, h










