import cv2
import numpy as np

from points_to_segments.hough import image_to_segments
from segments_to_polygons.main import segments_to_polygons, generate_position
from segments_to_polygons.visualization import visualize_segments, visualize_position
from segments_to_polygons.visualization import visualize_polygons


target_image = cv2.imread('imgs/bremen_altstadt_final.png')

position = generate_position(200, 800)
segments = image_to_segments(target_image)
polygons = segments_to_polygons(segments, position)

results_image = np.zeros(target_image.shape)
results_image = visualize_polygons(results_image, polygons)
results_image = visualize_segments(results_image, segments)
results_image = visualize_position(results_image, position)

print('position', position)

white_stripe = np.ones((target_image.shape[0], 5, 3)) * 255

results_image = np.hstack([target_image, white_stripe, results_image])

cv2.imwrite('bremen-polygons.png', results_image)
