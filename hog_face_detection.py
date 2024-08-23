from pyimagesearch.helpers import convert_and_trim_bb
import argparse
import imutils
import time
import dlib
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images",nargs='+', type=str, required=True, help="path(s) to input image(s)")
ap.add_argument("-d", "--directory", type=str, required=True, help="path to the folder containing images")
ap.add_argument("-u", "--upsample", type=int, default=1, help="# of times to upsample")

# Parse the arguments
args = ap.parse_args()

# Get the directory path
directory_path = args.directory
# Check if the directory exists
if not os.path.isdir(directory_path):
    print(f"Error: Directory {directory_path} does not exist.")
    exit()

# load dlib's HOG + Linear SVM face detector
print("[INFO] loading HOG + Linear SVM face detector...")
detector = dlib.get_frontal_face_detector()
# load the input image from disk, resize it, and convert it from
# BGR to RGB channel ordering (which is what dlib expects)
for filename in os.listdir(directory_path):
	file_path = os.path.join(directory_path, filename)
	if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
		image = cv2.imread(file_path)
		if image is None:
			print(f"Error: Could not load image {file_path}")
			continue
		image = imutils.resize(image, width=600)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# perform face detection using dlib's face detector
		start = time.time()
		print("[INFO[ performing face detection with dlib...")
		rects = detector(rgb, args.upsample)
		end = time.time()
		print("[INFO] face detection took {:.4f} seconds".format(end - start))

		# convert the resulting dlib rectangle objects to bounding boxes,
		# then ensure the bounding boxes are all within the bounds of the
		# input image
		boxes = [convert_and_trim_bb(image, r) for r in rects]
		# loop over the bounding boxes
		for (x, y, w, h) in boxes:
			# draw the bounding box on our image
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# show the output image
		cv2.imshow("Output", image)
		cv2.waitKey(0)

