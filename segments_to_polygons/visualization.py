import numpy as np
import cv2

from config import CANVAS_SIZE

GREEN = (0, 200, 0)
RED = (0, 0, 200)
WHITE = (255, 255, 255)


def visualize_segments(image, segments):
  '''
  image: numpy array
  segments: [n, 4] matrix
  '''
  for segment in segments:
    segment = np.array(segment).astype(np.uint32)
    cv2.line(image, tuple(segment[:2]), tuple(segment[2:]), WHITE, 3)

  cv2.imwrite('results/segment-visualization.png', image)
  return image

def visualize_position(image, position):
  cv2.circle(image, tuple(position), 6, RED, -1)
  cv2.circle(image, tuple(position), 14, RED, 2)
  cv2.imwrite('results/with-position.png', image)
  return image

def visualize_polygons(image, polygons):
  for polygon in polygons:
    new_polygon = np.clip(np.array(polygon), 1, CANVAS_SIZE-1).astype(int)
    try:
      cv2.fillConvexPoly(image, new_polygon, GREEN)
    except:
      print('Throwing out bad polygon.')

  cv2.imwrite('results/with-polygons.png', image)
  return image
