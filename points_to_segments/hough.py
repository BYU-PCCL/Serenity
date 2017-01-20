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
		lines = cv2.HoughLines(edges,1,np.pi/180,150)

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
			cv2.line(self.image,line[0],line[1],(0,0,255),2)
		
		cv2.imwrite('houghlines.jpg',self.image)

def main():
	h = Hough(imageName='sudoku-original.jpg')
	edges = h.TransformImage()
	print edges

	segments = h.GetLineSegments(edges)
	h.DrawSegments(segments)

if __name__ == '__main__':
    main()		



