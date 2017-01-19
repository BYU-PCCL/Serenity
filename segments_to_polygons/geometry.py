from math import cos, sin

import numpy as np

def linear_interpolate(point_a, point_b, proportion):
  return point_a + proportion * (point_b - point_a)

def shorten_point(anchor_point, point_to_be_shortened, keep_proportion):
  '''
  Returns a new point somewhere between anchor_point and point_to_be_shortened
  '''
  return (
    linear_interpolate(anchor_point[0], point_to_be_shortened[0], keep_proportion),
    linear_interpolate(anchor_point[1], point_to_be_shortened[1], keep_proportion)
  )

def point_distance(point_a, point_b):
  '''
  Returns the Euclidean distance between two points.
  '''
  return np.sqrt(np.sum((point_a-point_b)**2))

def farthest_point(anchor, points):
  '''
  Returns the farthest point in `points` relative to `anchor`.
  '''
  longest_point = None
  longest_distance = float('-inf')

  for point in points:
    distance = point_distance(anchor, point)
    if distance > longest_distance:
      longest_distance = distance
      longest_point = point

  return longest_point

def rotate_point_about_point(point, anchor, radians):
  '''
  Rotate `point`, treating `anchor` as the origin.
  '''
  point_x, point_y = point
  anchor_x, anchor_y = anchor

  new_point_x = cos(radians) * (point_x-anchor_x) - sin(radians) * \
  (point_y-anchor_y) + anchor_x
  new_point_y = sin(radians) * (point_x-anchor_x) + cos(radians) * (point_y-anchor_y) + anchor_y

  return new_point_x, new_point_y
