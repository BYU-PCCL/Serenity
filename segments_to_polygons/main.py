from __future__ import division, print_function

from math import pi

import numpy as np

from config import *
from visualization import visualize_segments, visualize_position
from visualization import visualize_polygons
from geometry import point_distance, rotate_point_about_point
from geometry import shorten_point, farthest_point

NINETY_DEGREES = pi/2

def generate_segments(number_of_segments):
  '''
  Segment format:
  x1 y1 x2 y2
  x1 y1 x2 y2

  origin top left, from 0 <= x < 1000
  '''
  return np.random.random_integers(
    GENERATION_LOWER_BOUND,
    GENERATION_UPPER_BOUND,
    size=(number_of_segments, 4)
  )

def generate_position(lower_bound=GENERATION_LOWER_BOUND, upper_bound=GENERATION_UPPER_BOUND):
  '''
  Returns an array of two numbers.
  '''
  return np.random.random_integers(lower_bound, upper_bound, size=2)

def get_rear_rectangle_corners(segment, position):
  '''
  Determine where the corners are that are farther from `position`. Rotates the 
  bottom about the top 90 degrees, and a second time by -90 degrees. Takes the
  farthest point as one corner, and produces the other corner.
  '''
  segment_top, segment_bottom = segment[:2], segment[2:]

  rotated_a = rotate_point_about_point(segment_bottom, segment_top, -NINETY_DEGREES)
  rotated_b = rotate_point_about_point(segment_bottom, segment_top, NINETY_DEGREES)
  corner1 = farthest_point(position, [rotated_a, rotated_b])

  correct_angle = NINETY_DEGREES if corner1 == rotated_a else -NINETY_DEGREES
  corner2 = rotate_point_about_point(segment_top, segment_bottom, correct_angle)

  return corner1, corner2

def segments_to_polygons(segments, position, vision_range=200):
  '''
  segments: an matrix of shape [n, 4] where n is the number of segments.
  position: a 2-element array

  Returns polygons: a list (all polygons) of lists (individual polygons) of 
  pairs (individual points).
  '''
  polygons = []

  for segment in segments:
    assumed_area = np.random.normal(
      loc=BUILDING_SIZE, 
      scale=(10 * BUILDING_VARIANCE)**2
    )

    segment_top, segment_bottom = segment[:2], segment[2:]

    face_length = point_distance(segment_top, segment_bottom)
    # area = length * width, solve for length
    side_length = min(assumed_area / face_length, MAX_BUILDING_LENGTH)
    proportion_to_keep = side_length / face_length

    corner1, corner2 = get_rear_rectangle_corners(segment, position)

    if face_length < 10 or \
      (point_distance(position, corner1) > vision_range and \
       point_distance(position, corner2) > vision_range):
       continue

    polygons.append([
      segment_bottom,
      shorten_point(segment_bottom, corner2, proportion_to_keep),
      shorten_point(segment_top, corner1, proportion_to_keep),
      segment_top
    ])

  return polygons

###############################################################################

def demo():
  '''
  Uses randomly generated segments and position.
  '''
  image = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3)) + 50
  segments = generate_segments(NUMBER_OF_SEGMENTS)
  position = generate_position()

  polygons = segments_to_polygons(segments, position)

  image = visualize_polygons(image, polygons)
  image = visualize_segments(image, segments)
  image = visualize_position(image, position)

if __name__ == '__main__':
  demo()
