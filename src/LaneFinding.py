import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image and print out some stats
Image = mpimg.imread("../res/Test.jpg")
print('This image is: ', type(Image), 'with dimensions: ', Image.shape)

# Grab the x and y size and make a copy of the image
Ysize = Image.shape[0]
Xsize = Image.shape[1]

# Note: always make a copy rather than simply using "=" 
ColorSelect = np.copy(Image)

# Define a triangle region of interest 
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
LeftBottom = [150, 539]
RightBottom = [850, 539]
Apex = [500, 250]

# Fit lines (y=Ax+B) to identify the  3 sided region of interest
# np.polyfit() returns the coefficients [A, B] of the fit
FitLeft = np.polyfit((LeftBottom[0], Apex[0]), (LeftBottom[1], Apex[1]), 1)
FitRight = np.polyfit((RightBottom[0], Apex[0]), (RightBottom[1], Apex[1]), 1)
FitBottom = np.polyfit((LeftBottom[0], RightBottom[0]), (LeftBottom[1], RightBottom[1]), 1)

# Define our color selection criteria
RedThreshold = 200
GreenThreshold = 200
BlueThreshold = 200
RgbThreshold = [RedThreshold, GreenThreshold, BlueThreshold]

# Identify pixels below the threshold
ColorThresholds = (Image[:, :, 0] < RgbThreshold[0])\
            | (Image[:, :, 1] < RgbThreshold[1])\
            | (Image[:, :, 2] < RgbThreshold[2])
            
# Find the region inside the lines
# Left(Ax+B < y), Right(Ax+B < y), Bottom(Ax+B > y)
XX, YY = np.meshgrid(np.arange(0, Xsize), np.arange(0, Ysize))
RegionThreshold = (YY > (XX*FitLeft[0] + FitLeft[1])) & \
                    (YY > (XX*FitRight[0] + FitRight[1])) & \
                    (YY < (XX*FitBottom[0] + FitBottom[1]))

# Mask color selection
ColorSelect[ColorThresholds] = [0, 0, 0] 

# Find where image is both colored right and in the region
ColorSelect[~ColorThresholds & RegionThreshold] = [255, 0, 0]

# Display the image
plt.imshow(ColorSelect)
plt.show()