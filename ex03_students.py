
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import signal
from IP_ex3_func import sobel, non_max_suppresion, colorTransform, convolution

plt.close('all')
cv2.destroyAllWindows()

# %% 1.a
# test points image 01
image_ptsInRow = np.zeros((40, 40))
image_ptsInRow[5, 0] = 1
image_ptsInRow[15, 10] = 1
image_ptsInRow[25, 20] = 1
image_ptsInRow[35, 30] = 1

# test points image 02
image_4pts = np.zeros((40, 40))
image_4pts[20, 7] = 1
image_4pts[7, 20] = 1
image_4pts[20, 33] = 1
image_4pts[33, 20] = 1

# test points image 03
image_noise = np.zeros((40, 40))
image_noise[np.random.randint(37, size=(23, 1)),
            np.random.randint(37, size=(23, 1))] = 1

# test lines image 04
image_lines = np.zeros((40, 40))
image_lines[:, 15] = 1
for ii in range(1, 40, 1):
    image_lines[ii, -ii] = 1
    

# %% 1.b
# Edge detection
# test images are already binary images
fig, axes = plt.subplots(2,2)
axes[0,0].imshow(image_ptsInRow) # 1s and 0s only
axes[0,0].set_title('pts in row')
axes[0,1].imshow(image_4pts)
axes[0,1].set_title('4 pts')
axes[1,0].imshow(image_noise)
axes[1,0].set_title('noise')
axes[1,1].imshow(image_lines)
axes[1,1].set_title('lines')
plt.tight_layout()
plt.savefig('original_images.png')


# Initialization accumulator (2D histogram)
# TODO
def hough_transform(image, step_width):
    alpha = np.arange(-90,90,step_width)  # TODO
    alpha = np.deg2rad(alpha) # convert from deg to rad
    
    # Voting histogram (2D) using all edge pixels by accumulating all the same points
    # get each edge pixel
    y_len, x_len = np.shape(image)  # (row, col) indexes to edges (loc); skip edge detection for test images
    max_d = int(np.sqrt(y_len**2+x_len**2)) #+ 100 # TODO; maximum possible distance is the diagonal length of an image
    all_d = np.linspace(-max_d, max_d, max_d*2) # 2x resolution
    accumulator = np.zeros((max_d*2, np.size(alpha) // step_width), dtype=np.uint64) # no 2*max_d
    
    y_pixels, x_pixels = np.nonzero(image)  # (row, col) indexes to edges (loc); skip edge detection for test images
    # loop over each pixel
    for i in range(len(x_pixels)):
        # TODO
        xi = x_pixels[i]
        yi = y_pixels[i]
        
        for idx_alpha in range(0, alpha.size):
            # TODO
            alpha_i = alpha[idx_alpha]
            d = int(xi * np.cos(alpha_i) + yi * np.sin(alpha_i)) # expressed in the new parametric parameters
            accumulator[d, idx_alpha] += 1 # 0 to 180deg in angle variation
    
    return accumulator, alpha, all_d

def invert_hough_transform(alpha_i, d_i, len_d):
    # # Method 2 : y = mx+c => psin(theta) = m(pcos(theta)) + c
    a = np.cos(alpha_i)
    b = np.sin(alpha_i)
    x0 = a * d_i
    y0 = b * d_i
    x1 = x0 + len_d * (-b) 
    y1 = y0 + len_d * (a) # int(x0); -ve to account for -y axis direction
    x2 = x0 - len_d * (-b)
    y2 = y0 - len_d * (a) # for both sided extension; either direction doesn't matter
        
    return [x1, y1, x2, y2]

def getLines(hough_image, threshold):
    ls = []
    stepT = float(hough_image.shape[1]) / np.pi
    stepD = float( int(hough_image.shape[0]) )/2.0

    t=0; d=0
    for m in np.nditer(hough_image[:,:],op_flags =['readwrite']):
        if m > threshold:
            ls.append((float(d)/stepD, float(t)/stepT))
        t+=1
        if t >= hough_image.shape[1]:
            t=0
            d+=1

    return ls

def find_local_maxima(accumulator, threshold, neighborhood_size):
    local_maxima = []

    for i in range(accumulator.shape[0]):
        for j in range(accumulator.shape[1]):
            if accumulator[i, j] > threshold and is_local_maximum(accumulator, i, j, neighborhood_size):
                local_maxima.append((i, j)) # indices only

    return local_maxima

def is_local_maximum(accumulator, row, col, neighborhood_size):
    neighborhood = accumulator[
        max(0, row - neighborhood_size): min(accumulator.shape[0], row + neighborhood_size + 1),
        max(0, col - neighborhood_size): min(accumulator.shape[1], col + neighborhood_size + 1)
    ]

    return accumulator[row, col] == np.max(neighborhood)

image_current = image_ptsInRow # image_lines
x_len, y_len = np.shape(image_current)
step_width = 1 # resolution is every angle
[accumulator_ptsInRow, alpha_ptsInRow, all_d_ptsInRow] = hough_transform(image_ptsInRow, step_width) # return histogram value (num of intercepts)
[accumulator_4pts, alpha_4pts, all_d_4pts] = hough_transform(image_4pts, step_width)
[accumulator_noise, alpha_noise, all_d_noise] = hough_transform(image_noise, step_width)
[accumulator_lines, alpha_lines, all_d_lines] = hough_transform(image_lines, step_width)

fig, axes = plt.subplots(2,2)
axes[0,0].imshow(accumulator_ptsInRow)
axes[0,0].set_title('pts in row')
axes[0,1].imshow(accumulator_4pts)
axes[0,1].set_title('4 pts')
axes[1,0].imshow(accumulator_noise)
axes[1,0].set_title('noise')
axes[1,1].imshow(accumulator_lines)
axes[1,1].set_title('lines')
plt.tight_layout()
plt.savefig('2d_histogram_all.png')


# Define constants
neighborhood_size = 5 # 20 # define neighborhood size for searching local maxima
len_d = 500 # for drawing lines; #np.max(d_i)/2 # length of extended lines (500 ~ 1000 depending on image)

fig, axes = plt.subplots(2,2)
# # ptsInRow image
local_max_ptsInRow = find_local_maxima(accumulator_ptsInRow/accumulator_ptsInRow.max(), 0.499, neighborhood_size)

for d_idx, alpha_idx in local_max_ptsInRow: # from the accumulator matrix definition
    alpha_i = alpha_ptsInRow[alpha_idx]
    d_i = all_d_ptsInRow[d_idx] + np.max(all_d_ptsInRow) # image_accumulator[d_idx,alpha_idx]
    print('pts in row: ' + str(np.rad2deg(alpha_i)), str(d_i))
    [x1,y1,x2,y2] = invert_hough_transform(alpha_i, d_i, len_d) # return d and alpha values corresponding to dominant lines
    
    # draw line into original image
    # print(pt1,pt2)
    axes[0,0].plot([x1,x2], [y1,y2], color='blue', linewidth=2) # [x0,y0],[x1,y1] instead of [x0,x1],[y0,y1]

# # 4pts image
local_max_4pts = find_local_maxima(accumulator_4pts/accumulator_4pts.max(), 0.5, neighborhood_size)

for d_idx, alpha_idx in local_max_4pts: # from the accumulator matrix definition
    alpha_i = alpha_4pts[alpha_idx]
    d_i = all_d_4pts[d_idx] + np.max(all_d_4pts) # image_accumulator[d_idx,alpha_idx]
    print('4 pts: ' + str(np.rad2deg(alpha_i)), str(d_i))
    [x1,y1,x2,y2] = invert_hough_transform(alpha_i, d_i, len_d) # return d and alpha values corresponding to dominant lines
    
    # draw line into original image
    # print(pt1,pt2)
    axes[0,1].plot([x1,x2], [y1,y2], color='blue', linewidth=2)

# # noise image
local_max_noise = find_local_maxima(accumulator_noise/accumulator_noise.max(), 0.5, neighborhood_size)

for d_idx, alpha_idx in local_max_noise: # from the accumulator matrix definition
    alpha_i = alpha_noise[alpha_idx]
    d_i = all_d_noise[d_idx] + np.max(all_d_noise) # image_accumulator[d_idx,alpha_idx]
    # print('noise: ' + str(np.rad2deg(alpha_i)), str(d_i))
    [x1,y1,x2,y2] = invert_hough_transform(alpha_i, d_i, len_d) # return d and alpha values corresponding to dominant lines
    
    # draw line into original image
    # print(pt1,pt2)
    axes[1,0].plot([x1,x2], [y1,y2], color='blue', linewidth=2)
    
# # lines image
local_max_lines = find_local_maxima(accumulator_lines/accumulator_lines.max(), 0.5, neighborhood_size)

for d_idx, alpha_idx in local_max_lines: # from the accumulator matrix definition
    alpha_i = alpha_lines[alpha_idx]
    d_i = all_d_lines[d_idx] + np.max(all_d_lines) # image_accumulator[d_idx,alpha_idx]
    print('lines: ' + str(np.rad2deg(alpha_i)), str(d_i))
    [x1,y1,x2,y2] = invert_hough_transform(alpha_i, d_i, len_d) # return d and alpha values corresponding to dominant lines
    
    # draw line into original image
    # print(pt1,pt2)
    axes[1,1].plot([x1,x2], [y1,y2], color='blue', linewidth=2)


axes[0,0].imshow(image_ptsInRow) # 1s and 0s only
axes[0,0].set_title('pts in row')
axes[0,1].imshow(image_4pts)
axes[0,1].set_title('4 pts')
axes[1,0].imshow(image_noise)
axes[1,0].set_title('noise')
axes[1,1].imshow(image_lines)
axes[1,1].set_title('lines')
plt.tight_layout()
plt.savefig('overlayed_images.png')

# exit(0)
# %% 2.a
# Process real-world street image
image_street = plt.imread("building.jpg") # original 200x300x3
image_gray = cv2.cvtColor(image_street, cv2.COLOR_RGB2GRAY) # cv function 200x300
xlen, ylen = np.shape(image_gray)

kernel_gray = np.ones((3, 3)) * (1/3)
image_gray2 = colorTransform(image_street, kernel_gray) # self implement 200x300x3

fig, axes = plt.subplots(2,2)
axes[0,0].imshow(image_street)
axes[0,0].set_title('original')
axes[1,0].imshow(image_gray,'gray')
axes[1,0].set_title('CV gray output')
axes[1,1].imshow(image_gray2)
axes[1,1].set_title('kernel gray output')
fig.delaxes(axes[0,1])
plt.tight_layout()
plt.savefig('colour_transform.png')

# Define constants
edge_neighbour = 5 # 20 # edge neighbour size for preprocessing i.e. non-max suppression and Canny

# TODO
# need to do edge detection first (i.e. Sobel)
[image_dx,image_dy,image_dir,image_mag] = sobel(image_gray) # image_gray2
fig, axes = plt.subplots(2,2)
axes[0,0].imshow(image_dx)
axes[0,0].set_title('dx')
axes[0,1].imshow(image_dy)
axes[0,1].set_title('dy')
axes[1,0].imshow(image_mag)
axes[1,0].set_title('magnitude')
axes[1,1].imshow(image_dir)
axes[1,1].set_title('direction')
plt.tight_layout()
plt.savefig('sobel_out.png')

# Perform non_max suppression to improve contrast
image_edge = non_max_suppresion(image_dir,image_mag, edge_neighbour) # self implemented edge detection
image_edge2 = cv2.Canny(image_gray, 50, 150, edge_neighbour) # cv2 apply edge detection; can apply any kind e.g. Canny; (inputImage, minVal, maxVal)

fig, axes = plt.subplots(1,2,figsize=(8, 4))
axes[0].imshow(image_edge)
axes[0].set_title('non-max suppression')
axes[1].imshow(image_edge2)
axes[1].set_title('canny') # include non-max and other thresholding steps
plt.savefig('edge_detected_out.png')


# %% 2.b
# Self implemented Hough transform to extract prominent lines
# Define constants
local_neighbour = 5 # 20 # define local maxima neighborhood size for searching local maxima (again)

image_street = plt.imread("building.jpg") # original 200x300x3
threshold_percent = 0.47 # 0.27 # specific
[image_accumulator_nms, image_alpha_nms, all_d_nms] = hough_transform(image_edge, step_width) # self-implemented; non-max suppression
local_maxima_nms = find_local_maxima(image_accumulator_nms/image_accumulator_nms.max(), threshold_percent, local_neighbour)

for d_idx, alpha_idx in local_maxima_nms: # from the accumulator matrix definition
    alpha_i = image_alpha_nms[alpha_idx]
    d_i = all_d_nms[d_idx] + np.max(all_d_nms) # image_accumulator[d_idx,alpha_idx]
    # print([d_i, alpha_i])

    [x1,y1,x2,y2] = invert_hough_transform(alpha_i, d_i, len_d)
    
    # draw line into original image
    cv2.line(image_street, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 2)
    # plt.plot([x1, x2], [y1, y2], color='yellow', linewidth=2)    
    
fig, axes = plt.subplots(1,2,figsize=(8, 4))
axes[0].imshow(image_street)
axes[0].set_title('(NMS) Hough Lines, threshold = ' + str(threshold_percent)) # original size is (300,200)


image_street2 = cv2.imread("building.jpg") # original 200x300x3; repeat again
threshold_percent = 0.7 # specific
[image_accumulator_canny, image_alpha_canny, all_d_canny] = hough_transform(image_edge2, step_width) # self-implemented; canny
local_maxima_canny = find_local_maxima(image_accumulator_canny/image_accumulator_canny.max(), threshold_percent, local_neighbour)

for d_idx, alpha_idx in local_maxima_canny: # from the accumulator matrix definition
    alpha_i = image_alpha_canny[alpha_idx]
    d_i = all_d_canny[d_idx] + np.max(all_d_canny) # image_accumulator[d_idx,alpha_idx]
    # print([d_i, alpha_i])

    [x1,y1,x2,y2] = invert_hough_transform(alpha_i, d_i, len_d)
    
    # draw line into original image
    cv2.line(image_street2, (int(x1),int(y1)), (int(x2),int(y2)), (0, 0, 255), 2) # iverlay on another window
    # plt.plot([x1, x2], [y1, y2], color='yellow', linewidth=2)
image_street2 = cv2.cvtColor(image_street2, cv2.COLOR_BGR2RGB)
axes[1].imshow(image_street2)
axes[1].set_title('(Canny) Hough Lines, threshold = ' + str(threshold_percent)) # original size is (300,200)

# combine outputs for saving
plt.tight_layout()
plt.savefig('image_street_out.png')
