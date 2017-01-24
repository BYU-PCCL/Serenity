import cv2
import numpy as np

class Hough:

	def __init__(self, points=None, imageName=None):
		self.points = points
		self.imageName = imageName
		self.image = None

	def TransformImage(self):
		self.image = cv2.imread(self.imageName)
		gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray,50,150,apertureSize = 3)
		return edges

	def GetLineSegments(self, edges):
		segments = []
		lines = cv2.HoughLines(edges,1,np.pi/180,100)
		print "Lines:", lines
		for line in lines:
			rho = line[0][0]
			theta = line[0][1]
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))

			segments.append([(x1, y1), (x2, y2)])

		return segments

	def DrawSegments(self, segments):
		for line in segments:
			cv2.line(self.image,line[0],line[1],(255,0,0),2)
		
		cv2.imwrite('houghlines-test.png',self.image)


def main():

	img = cv2.imread('../point_clouds/bremen_altstadt_final.png')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,50,150,apertureSize = 3)
	minLineLength = 1000
	maxLineGap = 10
	lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)
	#print lines
	for line in lines:
		x1 = line[0][0]
		y1 = line[0][1]
		x2 = line[0][2]
		y2 = line[0][3]
		cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

	cv2.imwrite('houghlines5.jpg',img)`	q1

	# h = Hough(imageName='../point_clouds/bremen_altstadt_final.png')
	# edges = h.TransformImage()
	# print edges

	# segments = h.GetLineSegments(edges)
	# h.DrawSegments(segments)

if __name__ == '__main__':
    main()		



