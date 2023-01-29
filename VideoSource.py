from threading import Thread
from Queue import Queue
import cv2


class VideoGet:

	def __init__(self, src=0):
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		self.stopped = False
		self.Q = Queue(maxsize=1024)

	def start(self):
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		while not self.stopped:
			if not self.Q.full():
				self.grabbed, frame = self.stream.read()
				if not self.grabbed:
					print 'stopping the video source thread'
					self.stop()
					return
				self.Q.put(frame)

	def get(self):
		return self.Q.get()

	def more(self):
		return self.Q.qsize() > 0

	def stop(self):
		self.stopped = True
		self.Q.put(None)
		self.stream.release()
