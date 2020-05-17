#doing all the relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in the image and convert to grayscale
Image = mpimg.imread("res/exit-ramp.jpg")
Gray = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)

# Define a kernel size for Gaussian smoothing / blurring
KernalSize = 5
BlurGray = cv2.GaussianBlur(Gray, (KernalSize, KernalSize), 0)

# Define parameters for Canny and run it
LowThreshold = 50
HighThreshold = 150
Edges = cv2.Canny(BlurGray, LowThreshold, HighThreshold)

# Next we'll create a masked edges image using cv2.fillPoly()
Mask = np.zeros_like(Edges)
IgnoreMaskColor = 255

# This time we are defining a four sided polygon to mask
IMShape = Image.shape
Vertices = np.array([[(0,IMShape[0]),(450, 290), (490, 290), (IMShape[1],IMShape[0])]], dtype=np.int32)
cv2.fillPoly(Mask, Vertices, IgnoreMaskColor)
MaskedEdges = cv2.bitwise_and(Edges, Mask)

# Define the Hough transform parameters
Rho = 2
Theta = np.pi/180
Threshold = 15
MinLineLength = 40
MaxLineGap = 20

# Make a blank the same size as our image to draw on
Image = np.copy(Image) * 0

# Run Hough on edge detected image
Lines = cv2.HoughLinesP(MaskedEdges, Rho, Theta, Threshold, np.array([]), 
                        MinLineLength, MaxLineGap)

# Iterate over the output "lines" and draw lines on the blank
for Line in Lines:
    for x1, y1, x2, y2 in Line:
        cv2.line(Image, (x1, y1), (x2, y2), (255, 0, 0), 10)

# Create a "color" binary image to combine with line image
ColorEdges = np.dstack((Edges, Edges, Edges))

# Draw the lines on the edge image
Combo = cv2.addWeighted(ColorEdges, 0.8, Image, 1, 0)

# Display the image
plt.imshow(Combo)
