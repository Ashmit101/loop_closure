import cv2
import numpy as np
import pyfbow as bow
import os

k = 10
L = 6
nthreads = 1
maxIters = 0
verbose = False

WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080//2
WINDOW_NAME = 'Matched Image'

detector = cv2.ORB_create(nfeatures=200)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

print('Reading database')
voc = bow.Vocabulary(k, L, nthreads, maxIters, verbose)
voc.readFromFile('vocabulary.bin')

print('Reading images')
descriptors = []
keypoints = []

def compute_features(frame):
    gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(gr, mask=None)
    return des

def matchFeature(description):
    differences = []

    for des in descriptors:
        matchesList = bf.match(des, description)
        matchesList = sorted(matchesList, key = lambda x:x.distance)[:10]
        distances = []
        for matches in matchesList:
            if matches:
                distance = matches.distance
                distances.append(distance)
        average = sum(distances) / len(distances)
        differences.append(average)
    
    minimum_distance = min(differences)
    index_of_minimum = differences.index(minimum_distance)
    return index_of_minimum

def drawMatches(image, indexOfImageFromDataset):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp1, des1 = detector.detectAndCompute(image, mask=None)
    kp2, des2 = keypoints[indexOfImageFromDataset], descriptors[indexOfImageFromDataset]
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    imageFromDataset = cv2.imread(f'./data/my_frame_{indexOfImageFromDataset}.png')
    
    matchedImage = cv2.drawMatches(imageFromDataset, kp2, image, kp1, matches[:10], None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.resize(matchedImage, (WINDOW_WIDTH * 2, WINDOW_HEIGHT))
    cv2.imshow(WINDOW_NAME, matchedImage)

# Create list of descriptors of all images
image_files = os.listdir('./data')
for image_file in image_files:
    image_path = os.path.join('./data', image_file)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoint, description = detector.detectAndCompute(image, mask=None)
    descriptors.append(description)
    keypoints.append(keypoint)

# Read frame from video
video_capture = cv2.VideoCapture('my-video.mp4')

if (not video_capture.isOpened()):
    print("Error opening video stream of file")
    exit()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

while(video_capture.isOpened()):
    ret, frame = video_capture.read()
    if ret:
        # cv2.imshow(WINDOW_NAME, frame)
        description = compute_features(frame) # Extract feature from frame
        index_of_matched_image = matchFeature(description)

        drawMatches(frame, index_of_matched_image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break