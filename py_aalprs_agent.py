from lp_detector import LP_Detector
from lp_segmentator import LP_Segmentator
from char_classifier import Char_Classifier
from multiprocessing import Queue
from vehicle_detector import Vehicle_Detector


def main():
	print 'openning the capture file'
	work_queue = Queue()
	vehicle_lp_queue = Queue()
	lp_seqmentation_queue = Queue()
	segmentation_ocr_queue = Queue()
	v = Vehicle_Detector(0.7,'data/vehicle-detector/yolov3-tiny.weights',
						 'data/vehicle-detector/yolov3-tiny.cfg',
						 'data/vehicle-detector/coco.data', work_queue, vehicle_lp_queue,'30secs.mp4')

	l = LP_Detector(0.55, 'data/lp-detector/wpod-net_update1.h5',vehicle_lp_queue, lp_seqmentation_queue)

	ls = LP_Segmentator(0.4, 'data/ocr2/ocr-net_final.weights',
					   'data/ocr2/ocr-net.cfg','data/ocr2/ocr-net.data',
					   lp_seqmentation_queue, segmentation_ocr_queue)
	cc = Char_Classifier(0.5,'data/char_classifier/model_ALPR.h5','data/char_classifier/classes.txt',segmentation_ocr_queue , None)

	print 'starting the char classifier '
	cc.start()
	print 'starting the lp segmentor'
	ls.start()
	print 'starting the lp detector'
	l.start()

	print 'starting the vehicle detector'
	v.run()

	l.join()
if __name__ == '__main__':
	main()
