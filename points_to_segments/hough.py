from __future__ import division
import cv2
import numpy as np

class Hough:

	def __init__(self, image, points=None):
		self.points = points
		self.imageName = imageName
		self.image = None

	def TransformImage(self):
		self.image = cv2.imread(self.imageName)
		gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray,50,150,apertureSize = 3)
		cv2.imwrite('bremen-edges.png', edges)
		return edges

	def GetLineSegments(self, edges):
		segments = []
		lines = cv2.HoughLinesP(edges, 1, np.pi/360, 70)

		for x1, y1, x2, y2 in lines[0]:
			# print('line', line)
			# rho = line[0][0]
			# theta = line[0][1]
			# a = np.cos(theta)
			# b = np.sin(theta)
			# x0 = a*rho
			# y0 = b*rho
			# x1 = int(x0 + 1000*(-b))
			# y1 = int(y0 + 1000*(a))
			# x2 = int(x0 - 1000*(-b))
			# y2 = int(y0 - 1000*(a))
			segments.append([(x1, y1), (x2, y2)])

		return self.FormatForPolygonConversion(segments)

	def DrawSegments(self, segments):
		segments_only_image = np.zeros(self.image.shape)

		for line in segments:
			cv2.line(segments_only_image,line[0],line[1],(0,200,0),1)
		
		cv2.imwrite('houghlines-bremen.jpg', segments_only_image)


################################################################################

def main():
	h = Hough(imageName='bremen_altstadt_final.png')
	# h = Hough(imageName='sudoku-original.jpg')
	edges = h.TransformImage()

	h.image[h.image < 108] = 0
	h.image[h.image >= 108] = 255

	segments = h.GetLineSegments(cv2.cvtColor(h.image.astype(np.uint8), cv2.COLOR_BGR2GRAY))
	# h.DrawSegments(segments)

	print('segments', segments)

if __name__ == '__main__':
    main()

################################################################################

def image_to_segments(image):
	RESOLUTION = 1
	NUMBER_OF_ANGLES = 180
	VOTES_REQUIRED_TO_COUNT_AS_A_LINE = 70

	image[image < 100] = 0
	image[image >= 100] = 255
	image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)

	segments = []
	lines = cv2.HoughLinesP(
		image, 
		RESOLUTION,
		np.pi/NUMBER_OF_ANGLES, 
		VOTES_REQUIRED_TO_COUNT_AS_A_LINE
	)

	return np.array([[x1, y1, x2, y2] for x1, y1, x2, y2 in lines[0]])
