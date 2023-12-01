import cv2
import numpy as np
import pyfbow as bow

TIME_INTERVAL = 1 # seconds
WINDOW_WIDTH = 1920//2
WINDOW_HEIGHT = 1080//2
WINDOW_NAME = 'Video'

k = 10
L = 6
nthreads = 1
maxIters = 0
verbose = False

video_capture = cv2.VideoCapture('my-video.mp4')

if (not video_capture.isOpened()):
    print("Error opening video stream of file")
    exit()

fps = video_capture.get(cv2.CAP_PROP_FPS)
print("FPS of video: ", fps)

seconds = 0
frame_index = 0
detector = cv2.ORB_create(nfeatures=2000)
voc = bow.Vocabulary(k, L, nthreads, maxIters, verbose)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

des_list = []

def compute_features(frame):
    gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(gr, mask=None)
    print("Detected {} features".format(len(des)))
    print(des)
    return des

while(video_capture.isOpened()):
    frame_id = int(fps*(seconds))
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = video_capture.read()

    if ret:
        width, height = frame.shape[1], frame.shape[0]
        #resized_frame = cv2.resize(frame, (width//2, height//2))
        cv2.imshow(WINDOW_NAME, frame)
        cv2.imwrite(f'./data/my_frame_{frame_index}.png', frame)
        frame_index += 1

        description = compute_features(frame)
        des_list.append(description)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

    seconds += TIME_INTERVAL

dess = np.vstack(des_list)
voc.create(dess)

voc.saveToFile("vocabulary.bin")


video_capture.release()
cv2.destroyAllWindows()
