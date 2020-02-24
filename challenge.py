import cv2
import numpy as np
import scipy.signal
import scipy.cluster
from matplotlib import pyplot as plt
import time


# mode = "train"
mode = "test"

trainvid = cv2.VideoCapture("data/" + mode + ".mp4")


#  (221, 280)           (414, 250)
#
#  (100, 356)           (540, 356)

cropdim = 200
box = np.array([[221, 280], [414, 280], [540, 356], [100, 356]], np.float32)
square = np.array([[0, 0], [cropdim - 1, 0], [cropdim - 1, cropdim - 1], [0, cropdim - 1]], np.float32)
transform = cv2.getPerspectiveTransform(box, square)

lastframe = None

# Normalizes a grayscale image to have stddev 50 centered at 127
def normImage(im):
    imf = im.astype(float)
    return np.clip((((imf - np.mean(imf)) / np.std(imf)) * 50) + 127, 0, 255).astype(np.uint8)

# Gets the Y offset of two images
def getYoffset(cur, prev):
    # Use Sobel filter to find features in each image
    sobel_cur = cv2.Sobel(cur, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=5)
    sobel_prev = cv2.Sobel(prev, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=5)

    best_corr = 0
    offset = 0

    # Sweep current and previous images across each other to find offset with highest match
    for i in range(0, 100):
        cur_trim = sobel_cur[i:]
        prev_trim = sobel_prev[:cur.shape[0]-i]
        corrmat = np.multiply(cur_trim, prev_trim)
        corr = corrmat.mean()

        if corr > best_corr:
            best_corr = corr
            offset = i

    return offset


frame_num = 0
speeds = []
start = time.time()
while (trainvid.isOpened()):
    ret, frame = trainvid.read()
    if ret == True:
        # Take a box on the ground and warp it so that sizes/distances stay constant
        cropskewframe = cv2.warpPerspective(frame, transform, (cropdim, cropdim))

        # Transform to grayscale, normalize color, and blur to get rid of artifacts
        grayscale = cv2.cvtColor(cropskewframe, cv2.COLOR_BGR2GRAY)
        normalized = normImage(grayscale)
        blurred = cv2.GaussianBlur(normalized, ksize=(21, 21), sigmaX=4)

        cv2.imshow("frame", frame)

        # Find pixel offset between this frame and the last
        if lastframe is not None:
            speed = getYoffset(blurred, lastframe)
            speeds.append(speed)

        lastframe = blurred

        frame_num += 1

    else:
        break
speeds = [speeds[0]] + speeds # Duplicate first item so lengths match
numspeeds = len(speeds)

print(str(time.time() - start) + " seconds to analyze video")
start = time.time()

# Remove bad values with clustering
pruned_speeds = []
for i in range(0, len(speeds)):
    nearby_speeds = np.array(speeds[max(0, i - 30) : min(i + 31, numspeeds)]).astype(np.float)
    clusters = scipy.cluster.vq.kmeans(nearby_speeds, 2)
    topcluster = max(clusters[0])
    botcluster = min(clusters[0])

    bimodal = topcluster / (botcluster + 0.01) > 5
    closer_to_top = abs(topcluster - speeds[i]) < abs(botcluster - speeds[i])

    if (bimodal and closer_to_top) or not bimodal:
        pruned_speeds.append(speeds[i])
    else:
        pruned_speeds.append(None)

print(str(time.time() - start) + " seconds to cluster")
start = time.time()

# Smooth missing data slightly and fill gaps
smooth_speeds = []
gaussian_stddev = 5
for i in range(0, numspeeds):
    num = 0
    denom = 0
    for j in range(max(0, i - 10), min(i + 11, numspeeds)):
        if pruned_speeds[j] is not None:
            multiplier = np.exp(-((i - j) ** 2) / (2 * gaussian_stddev))
            num += multiplier * pruned_speeds[j]
            denom += multiplier
    if denom == 0:
        smooth_speeds.append(-10)
    else:
        smooth_speeds.append(num / denom)

print(str(time.time() - start) + " seconds to smooth")
start = time.time()

# Low-pass filter
b, a = scipy.signal.butter(8, 0.01)
filtered_speeds = scipy.signal.filtfilt(b, a, smooth_speeds, padlen=150)

print(str(time.time() - start) + " seconds to filter")
start = time.time()

# Scale speeds to match units, remove negative speeds
scaled_speeds = (filtered_speeds * 0.596) * (filtered_speeds > 0)

if mode == "train":
    # Calculate loss
    labels = np.loadtxt("data/train.txt")
    mse = ((labels[:numspeeds] - scaled_speeds) ** 2).mean()
    print("MSE: " + str(mse))

    # Show scores
    plt.figure()
    plt.plot(range(numspeeds), scaled_speeds, range(numspeeds), labels[:numspeeds])
    plt.show()

elif mode == "test":
    np.savetxt("data/test.txt", scaled_speeds, fmt="%f")

    plt.figure()
    plt.plot(range(numspeeds), scaled_speeds)
    plt.show()
