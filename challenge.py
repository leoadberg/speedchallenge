import cv2
import numpy as np
import scipy.signal
import scipy.cluster
from matplotlib import pyplot as plt
import time

#  (263, 250)           (360, 250)
#  (221, 280)           (414, 250)
#
#
#  (100, 356)           (540, 356)

cropdim = 200

# box = np.array([[263, 377, 110, 530],
#                 [250, 250, 356, 356]])
box = np.array([[221, 280], [414, 280], [540, 356], [100, 356]], np.float32)

square = np.array([[0, 0], [cropdim - 1, 0], [cropdim - 1, cropdim - 1], [0, cropdim - 1]], np.float32)

transform = cv2.getPerspectiveTransform(box, square)

trainvid = cv2.VideoCapture("data/train.mp4")

lastframe = None

def normImage(im):
    imf = im.astype(float)
    return np.clip((((imf - np.mean(imf)) / np.std(imf)) * 50) + 127, 0, 255).astype(np.uint8)

frame_num = 0

labels = np.loadtxt("data/train.txt")
speeds = []

def getYoffset(cur, prev):
    # cmean = cur.mean()
    # pmean = prev.mean()
    # norm_cur = cur.astype(np.float) - cmean
    # norm_prev = prev.astype(np.float) - pmean

    cmean = 0
    pmean = 0
    norm_cur = cv2.Sobel(cur, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=5)
    norm_prev = cv2.Sobel(prev, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=5)


    best_corr = 0
    offset = 0
    for i in range(0, 100):
        cur_trim = norm_cur[i:]
        prev_trim = norm_prev[:cur.shape[0]-i]
        # from IPython import embed; embed()
        corrmat = np.multiply(cur_trim, prev_trim)
        # corr = np.tensordot(cur_trim, prev_trim, axes=((0,1),(0,1))) / cur_trim.size
        corr = corrmat.mean()
        # if i == 48:
        #     from IPython import embed; embed()
        # print(i, corr)
        # cv2.imshow('cur_trim', (cur_trim + cmean).astype(np.uint8))
        # cv2.imshow('prev_trim', (prev_trim + pmean).astype(np.uint8))
        # cv2.imshow('corr', (corrmat / np.std(corrmat) * 20 + 50).astype(np.uint8))
        # plt.figure("hist")
        # plt.hist(corrmat, bins="auto")
        # plt.show(block=False)

        # if cv2.waitKey() & 0xFF == ord('q'):
        #     break
        if corr > best_corr:
            best_corr = corr
            offset = i
    return offset


start = time.time()
while (trainvid.isOpened()):
    ret, frame = trainvid.read()
    if ret == True:


        cropskewframe = cv2.warpPerspective(frame, transform, (cropdim, cropdim))
        grayscale = cv2.cvtColor(cropskewframe, cv2.COLOR_BGR2GRAY)
        normalized = normImage(grayscale)
        blurred = cv2.GaussianBlur(normalized, ksize=(31, 31), sigmaX=4)
        # cropskewframe = np.zeros((cropdim, cropdim, 3), dtype=float)
        # from IPython import embed; embed()

        cv2.imshow('Frame', frame)

        if lastframe is not None:
            # diff = scipy.signal.fftconvolve(lastframe, normalized[::-1,::-1], mode='same')
            # diff = scipy.signal.oaconvolve(lastframe, normalized[::-1,::-1], mode='same')
            # diff = scipy.signal.fftconvolve(lastframe - lastframe.mean(), (normalized - normalized.mean())[::-1,::-1], mode='same')
            # diff = scipy.signal.correlate2d(lastframe - lastframe.mean(), normalized - normalized.mean(), boundary='symm', mode='same')
            # diffImage = normImage(diff)
            # cv2.imshow('Diff', diffImage)
            cv2.imshow('Old', lastframe)
            cv2.imshow('New', blurred)
            # cv2.imshow('Sobel', cv2.Sobel(blurred, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=5))
            # shift = np.unravel_index(np.argmax(diff), diff.shape)
            # shift = cv2.phaseCorrelate(lastframe.astype(float), blurred.astype(float))
            # print(shift)
            # if shift[1] > 0.5:
            # speeds.append(np.linalg.norm(shift[0]))
            speed = getYoffset(blurred, lastframe)
            # speed = 0
            # print(speed)
            speeds.append(speed)
            # else:
            #     speeds.append(None)
            # from IPython import embed; embed()

            # corners = cv2.cornerHarris(normalized, blockSize=10, ksize=3, k=0.04)
            # cv2.imshow('Corners', corners)

            # edges = cv2.Canny(normalized, 100, 255)
            # edges = cv2.Sobel(normalized, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=5)
            # cv2.imshow('Blurred', blurred)
            # from IPython import embed; embed()
        lastframe = blurred

        # if frame_num > 10000:
        #     if cv2.waitKey() & 0xFF == ord('q'):
        #         break
            # break
        frame_num += 1

        # Press Q on keyboard to  exit
        # if cv2.waitKey() & 0xFF == ord('q'):
        #   break

    else:
        break
numspeeds = len(speeds)

print(str(time.time() - start) + " seconds to parse video")
start = time.time()

# Remove bad values
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

# Smooth missing data and fill gaps
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

# Scale speeds to match truth
scaled_speeds = filtered_speeds * 0.583

# Calculate loss
mse = ((labels[:numspeeds] - scaled_speeds) ** 2).mean()
print("MSE: " + str(mse))

# Show scores
plt.figure()
plt.plot(range(numspeeds), scaled_speeds, range(numspeeds), labels[:numspeeds])
plt.show()
