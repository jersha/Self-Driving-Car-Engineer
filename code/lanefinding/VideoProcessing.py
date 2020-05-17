import numpy as np
import cv2
import matplotlib.pyplot as plt

def f_Gray(Image):
    return cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

def f_GaussianBlur(Image):
    KernalSize = 5      # Define a kernel size for Gaussian smoothing / blurring
    return cv2.GaussianBlur(Image, (KernalSize, KernalSize), 0)

def f_Canny(Image):
    LowThreshold = 50   # Define parameters for Canny 
    HighThreshold = 150 # Define parameters for Canny 
    return cv2.Canny(Image, LowThreshold, HighThreshold)

def f_Mask(Image):
     # Next we'll create a masked edges image using cv2.fillPoly()
    Mask = np.zeros_like(Image)
    IgnoreMaskColor = 255
    
    # This time we are defining a four sided polygon to mask
    IMShape = Image.shape
    Vertices = np.array([[(200, IMShape[0]), (600, 450), (800, 450), (1200, IMShape[0])]], dtype=np.int32)
    cv2.fillPoly(Mask, Vertices, IgnoreMaskColor)
    return cv2.bitwise_and(Image, Mask)

def f_Hough(Image, MaskedEdges):
    # Define the Hough transform parameters
    Rho = 2
    Theta = np.pi/180
    Threshold = 15
    MinLineLength = 40
    MaxLineGap = 20
    
    # Make a blank the same size as our image to draw on
    Frame = Image * 0
    
    # Run Hough on edge detected image
    Lines = cv2.HoughLinesP(MaskedEdges, Rho, Theta, Threshold, np.array([]), 
                            MinLineLength, MaxLineGap)
    
    # Iterate over the output "lines" and draw lines on the blank
    for Line in Lines:
        for x1, y1, x2, y2 in Line:
            cv2.line(Frame, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return Frame
            
Infile = cv2.VideoCapture('res/test.mp4')
Ret, Inframe = Infile.read()
print('ret =', Ret, 'W =', Inframe.shape[1], 'H =', Inframe.shape[0], 'channel =', Inframe.shape[2])
FPS= 20.0
FrameSize=(Inframe.shape[1], Inframe.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

Outfile = cv2.VideoWriter('Video_output.mp4', fourcc, FPS, FrameSize, 0)

while(Infile.isOpened()):
    Ret, Image = Infile.read()

    # check for successfulness of cap.read()
    if not Ret: break

    Gray = f_Gray(Image)
    BlurGray = f_GaussianBlur(Gray)
    Edges = f_Canny(BlurGray)
    MaskedEdges = f_Mask(Edges)
    Frame = f_Hough(Image, MaskedEdges)
    
    # Create a "color" binary image to combine with line image
    ColorEdges = np.dstack((Edges, Edges, Edges))
    
    # Draw the lines on the edge image
    Combo = cv2.addWeighted(ColorEdges, 0.8, Frame, 1, 0)

    cv2.imshow('frame',Combo)

    # Save the video
    Outfile.write(Combo)

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

Infile.release()
Outfile.release()
cv2.destroyAllWindows()