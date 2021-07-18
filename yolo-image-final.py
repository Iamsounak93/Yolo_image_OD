
# Importing needed libraries
import numpy as np
import cv2
import time


"""
Reading input image
"""
image_BGR = cv2.imread('images/2.png')

cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', image_BGR)

cv2.waitKey(0)
cv2.destroyWindow('Original Image')

# Getting spatial dimension of input image
h, w = image_BGR.shape[:2]

"""
Getting blob from input image
"""
blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),
                             swapRB=True, crop=False)

"""
Loading YOLO v3 network
"""
# for Windows, yours path might looks like:
# 'yolo-coco-data\\coco.names'
with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                     'yolo-coco-data/yolov3.weights')

layers_names_all = network.getLayerNames()

layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.5

# Setting threshold for filtering weak bounding boxes with non-maximum suppression
threshold = 0.3

# Generating colours for representing every detected object
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

"""
Implementing Forward pass
"""
# Implementing forward pass with our blob and only through output layers
# Calculating at the same time, needed time for forward pass
network.setInput(blob)  # setting blob as input to the network
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()

# Showing spent time for forward pass
print('Objects Detection took {:.5f} seconds'.format(end - start))

"""
Getting bounding boxes
"""

# Preparing lists for detected bounding boxes, obtained confidences and class's number
bounding_boxes = []
confidences = []
class_numbers = []

for result in output_from_network:
    for detected_objects in result:
        scores = detected_objects[5:]        # Getting 80 classes' probabilities for current detected object
        class_current = np.argmax(scores)    # Getting index of the class with the maximum value of probability
        confidence_current = scores[class_current]     # Getting value of probability for defined class

        # Eliminating weak predictions with minimum probability
        if confidence_current > probability_minimum:
            box_current = detected_objects[0:4] * np.array([w, h, w, h])   # Because YOLO data format keeps coordinates for center of bounding box and its current width and height

            x_center, y_center, box_width, box_height = box_current
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))

            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)

"""
Non-maximum suppression
"""

# Implementing non-maximum suppression of given bounding boxes
results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                           probability_minimum, threshold)

"""
Drawing bounding boxes and labels
"""

# Defining counter for detected objects
counter = 1

# Checking if there is at least one detected object after non-maximum suppression
if len(results) > 0:
    for i in results.flatten():
        print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))    # Showing labels of the detected objects
        counter += 1

        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        colour_box_current = colours[class_numbers[i]].tolist()

        cv2.rectangle(image_BGR, (x_min, y_min),
                      (x_min + box_width, y_min + box_height),
                      colour_box_current, 2)

        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                               confidences[i])                     # Preparing text with label and confidence for current bounding box

        cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

print()
print('Total objects been detected:', len(bounding_boxes))
print('Number of objects left after non-maximum suppression:', counter - 1)        # Comparing how many objects where before non-maximum suppression left

cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
cv2.imshow('Detections', image_BGR)
cv2.waitKey(0)
cv2.destroyWindow('Detections')
