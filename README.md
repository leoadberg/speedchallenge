# Analyzing speed using traditional CV

I decided to attempt the <a href="https://github.com/commaai/speedchallenge">comma.ai speed challenge</a> for fun (not looking for a job at the moment). The task seems relatively straightforward, and as all other attempts of the speed challenge use some form of neural network, I thought it would be interesting to use traditional computer vision techniques instead.

The challenge is to predict the speed of a car based solely on dashcam video. The training set is 17 minutes of driving with speed values for each frame, and the test set is 9 minutes of similar footage.

The results are great and match the labels pretty well. The MSE is 1 but taken with a grain of salt because there is no validation set. All parameters were tuned by hand on the first few hundred frames. The orange line is ground truth and the blue line is my prediction.

![Predicted vs truth](/pics/trainbig.png)

## Algorithm

### Frame -> Offset

- Crop out the road in front of the car and "unskew" it into a flat square
- Normalize the image to account for exposure changes
- Blur the image to hide video artifacts while leaving features intact
- Detect features using a Sobel filter
- Detect offset in pixels from previous frame's features by sweeping against each other and rating correlation

### Offsets -> Speeds

- Use clustering to remove outliers (bad offsets)
- Gaussian blur to fill in missing data (and smooth noise a bit)
- Low-pass filter to smooth out noise and provide realistic results (acceleration and jerk are limited in real life)
- Scale by constant factor to convert from pixel offsets to speed values

## Technical details

### Unskewing / Normalizing

This is relatively simple, I just use `cv2.warpPerspective()` to unskew the road in front of the car. Then, I convert to grayscale and set the mean of the image to 127 and standard deviation to 50:

![Original](/pics/frame.png)
![Unskewed](/pics/unskew.png)

### Feature Detection

I first apply a gaussian blur to the image to make sure no video artifacts that were enhanced when unskewing/normalizing show up. Then, I apply a Sobel filter to highlight all features in the image.

![Blurred](/pics/cur.png)
![Sobeled](/pics/sobelcur.png)

### Offset Detection

Using the features from the current frame and the previous, I sweep the two images against each other and measure correlation at each point. I do this by computing an element-wise multiply at the specified offset then getting the mean of the result. I then choose the offset with the highest correlation. As you can see, in this example the correlation is highest at a 25 pixel offset (when the arrows line up and "corrmat" is brightest).

![Correlation Sweep](/pics/correlation.gif)
![Correlation Graph](/pics/25pixsec.png)

### Speed Cleaning

The raw offsets are very noisy and contain bad values where there aren't enough features (orange is truth):

![Raw offsets](/pics/rawspeed.png)

Here is a zoomed in version to show what the data looks like in a ~10 seconds of video with few features:

![Zoomed](/pics/rawspeedzoomed.png)

To fix this, for each data point I cluster the surrounding data points using K-Means and remove if it's part of the outlier cluster. Then, I fill missing points with a normalized gaussian kernel and I use a low-pass Butterworth filter to remove noise. Finally, I scale by a constant to match the speed labels:

![Predicted vs truth](/pics/trainbig.png)

# Test results

My MSE on the test set is 13, much higher than the MSE of the training set. I don't believe my algorithm overfit on the training set as there are only a few tuned parameters, but that simply the training set and test set have different elements. As I designed the algorithm for the training set, it doesn't take into account some feature of the test set like:

- Sharp turns: The training video has zero sharp turns (the only turns are slow ones on the highway), but the test video has a bunch. I suspect that my algorithm fails on turns because it doesn't check for any offset in the horizontal direction when doing the correlation sweep.
- Cross traffic: The training video doesn't include any points where the car is stopped and there is cross traffic directly in front. However, this happens a few times in the test video, which probably messes up my algorithm. At the moment, I only detect outliers as being below the true speed, but these would create outliers higher than the true speed of 0 mph.

# Future Improvements

- Use only past data
  - It would be cool to modify the algorithm so that only previous data is needed to predict the current speed. This is not required by the challenge, but obviously would be needed in real life. The only parts that need changing would be the cleaning of speed values (clustering, lo-pass filtering, etc).
- Get speed by regressing with realistic limits
  - Instead of just using a low-pass filter, it would be cool to construct a curve that obeys limits of real car physics (acceleration, jerk, max speeds) and fit it to the raw data
- Get speed by finding the minimum-energy path through the correlations over time
  - A bit complex, but if you take the correlation for offsets [0, 100] for the first 1000 frames, you get an image that looks like this. (Horizontal axis is offset, vertical axis is time)
  - The minimum energy path from the top to the bottom of the image is the speed of the car.
  - This would allow for correction of previously incorrect offset estimates.

  ![Correlation over time](/pics/2dcorrelation.png)
