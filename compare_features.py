import cv2
import numpy as np
import pyfbow as bow

k = 10
L = 6
nthreads = 1
maxIters = 0
verbose = False

detector = cv2.ORB_create(nfeatures=2000)
print('Reading database')

voc = bow.Vocabulary(k, L, nthreads, maxIters, verbose)
voc.readFromFile('vocabulary.bin')

print('Reading images')
images = []

