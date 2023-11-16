import cv2
import sys
import numpy as np
from scipy.interpolate import *
import keyboard

# python ImageProcessing.py problem1 input1.jpg
def problem1(image):
    # Initialises image data
    img = cv2.imread(image).astype(np.float32)
    img_height = img.shape[0]
    img_width = img.shape[1]

    # Creates mask using polygon formed by pts
    pts = np.array([[40*img_width/100, 0], [52*img_width/100, 0], [82*img_width/100, img_height], [75*img_width/100, img_height]], np.int32)
    initial_mask = np.zeros_like(img)
    initial_mask = cv2.fillPoly(initial_mask, [pts], (1, 1, 1))

    # Called to by trackbars when they are changed
    def empty(x):
        pass

    # Initialize trackbar data
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("mode", "TrackBars", 0, 1, empty)
    cv2.createTrackbar("image_wht", "TrackBars", 75, 100, empty)
    cv2.createTrackbar("mask_wht", "TrackBars", 35, 100, empty)
    cv2.createTrackbar("blend_wht", "TrackBars", 35, 200, empty)
    cv2.createTrackbar("clr_shift", "TrackBars", 5, 360, empty)
    cv2.createTrackbar("clr_step", "TrackBars", 5, 20, empty)

    # While loop until Trackbars is closed
    while cv2.getWindowProperty("TrackBars", 0) >= 0:

        # Retrieves trackbar values
        mode = cv2.getTrackbarPos("mode", "TrackBars")
        image_weight = cv2.getTrackbarPos("image_wht", "TrackBars")
        mask_weight = cv2.getTrackbarPos("mask_wht", "TrackBars")
        blend_weight = cv2.getTrackbarPos("blend_wht", "TrackBars")
        blend_weight_mod = blend_weight % 2  # Ensures that even numbers aren't chosen for gaussian blur
        if blend_weight_mod == 0:
            blend_weight += 1
        rainbow_shift = cv2.getTrackbarPos("clr_shift", "TrackBars")
        rainbow_step = cv2.getTrackbarPos("clr_step", "TrackBars")

        # Performs gaussian blur on mask to remove harsh edges
        mask = cv2.GaussianBlur(initial_mask, (blend_weight, blend_weight), 0)
        # Multiplies mask with image
        mask = cv2.multiply(img, mask)

        if mode == 1:  # RAINBOW MODE
            # Converts mask to HSV and extracts hue, saturation and value
            mask_HSV = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(mask_HSV)

            # Creates colour gradient array
            gradient_array = [(i*rainbow_step) + rainbow_shift for i in range(0, img_width)]
            # Creates gradient matrix where every row is the gradient array
            gradient_matrix = [gradient_array for i in range(img_height)]

            # Assigns new hue to gradient matrix
            h = np.asmatrix(gradient_matrix, np.float32)
            # Gives saturation max value so colours are vivid
            s = np.full((img_height, img_width), 1, np.float32)

            # Creates new rainbow mask
            mask = cv2.merge([h, s, v])
            # Converts mask back to BGR
            mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

        # Adds mask to image with weighting
        image_component = image_weight * img / 100
        mask_component = mask_weight * mask / 100
        # Clips values to correct range
        result_img = np.clip(image_component + mask_component, 0, 255)

        # Shows result
        cv2.imshow("result_img", result_img.astype(np.uint8))
        cv2.waitKey(1)

        # Checks for 'p' press
        if keyboard.is_pressed('p'):
            cv2.imwrite("problem1.jpg", result_img.astype((np.uint8)))
    cv2.destroyAllWindows()

# python ImageProcessing.py problem2 input1.jpg
def problem2(image):
    # Initialises image data
    img = cv2.imread(image).astype(np.float32)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian noise
    noise = np.random.normal(127, 127, (img.shape[0], img.shape[1], 1)).astype(np.float32)
    noise2 = np.random.normal(127, 127, (img.shape[0], img.shape[1], 1)).astype(np.float32)

    # # Salt and pepper
    # noise = np.zeros_like(img)[:, :, 0]
    # noise2 = np.zeros_like(img)[:, :, 0]
    # probability = 0.5
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         if np.random.random() < probability:
    #             noise[i][j] = 255
    #         if np.random.random() < probability:
    #             noise2[i][j] = 255

    # Called to by trackbars when they are changed
    def empty(x):
        pass

    # Initialize trackbar data
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("mode", "TrackBars", 0, 1, empty)
    cv2.createTrackbar("noise_wht", "TrackBars", 30, 100, empty)
    cv2.createTrackbar("blur_size", "TrackBars", 5, 50, empty)
    cv2.createTrackbar("colour", "TrackBars", 0, 2, empty)

    # While loop until Trackbars is closed
    while cv2.getWindowProperty("TrackBars", 0) >= 0:

        # Retrieves trackbar values
        mode = cv2.getTrackbarPos("mode", "TrackBars")
        noise_weight = cv2.getTrackbarPos("noise_wht", "TrackBars")
        blur_size = cv2.getTrackbarPos("blur_size", "TrackBars")
        blur_size_mod = blur_size % 2  # Ensures that even numbers aren't chosen for motion blur
        if blur_size_mod == 0:
            blur_size += 1
        colour = cv2.getTrackbarPos("colour", "TrackBars")

        # Creates empty array of zeros
        motion_blur = np.zeros((blur_size, blur_size))
        # Makes diagonal row equal to 1
        for i in range(blur_size):
            motion_blur[i, blur_size - i - 1] = 1
        # Normalises kernel
        motion_blur = motion_blur / blur_size

        # Applies blur to noise
        noise_blur = cv2.filter2D(noise, -1, motion_blur)
        # Adds noise to image with weighting
        result_img = cv2.addWeighted(img_gray.astype(np.uint8), 1, noise_blur.astype(np.uint8), noise_weight / 100, 0)

        if mode == 1:
            # Applies blur to noise2
            noise_blur2 = cv2.filter2D(noise2, -1, motion_blur)
            # Creates 3 channel grayscale image
            channel_img = cv2.merge((img_gray, img_gray, img_gray))

            # Adds blurred noise to greyscale image
            c1 = cv2.addWeighted(img_gray.astype(np.uint8), 1, noise_blur.astype(np.uint8), noise_weight/100, 0)
            c2 = cv2.addWeighted(img_gray.astype(np.uint8), 1, noise_blur2.astype(np.uint8), noise_weight/100, 0)

            # Adds blurred greyscale to colour channels
            if colour == 0:
                channel_img[:, :, 0] = c1
                channel_img[:, :, 1] = c2
            if colour == 1:
                channel_img[:, :, 0] = c1
                channel_img[:, :, 2] = c2
            if colour == 2:
                channel_img[:, :, 1] = c1
                channel_img[:, :, 2] = c2

            # Assigns result to new image
            result_img = channel_img

        # Shows result
        cv2.imshow("result_img", result_img.astype(np.uint8))
        cv2.waitKey(1)

        # Checks for 'p' press
        if keyboard.is_pressed('p'):
            cv2.imwrite("problem2.jpg", result_img.astype((np.uint8)))
    cv2.destroyAllWindows()

# python ImageProcessing.py problem3 input1.jpg
def problem3(image):
    # Initialises image data
    img = cv2.imread(image).astype(np.float32)

    # Called to by trackbars when they are changed
    def empty(x):
        pass

    # Initialize trackbar data
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("mode", "TrackBars", 0, 2, empty)
    cv2.createTrackbar("krnl_size", "TrackBars", 20, 100, empty)
    cv2.createTrackbar("sigma", "TrackBars", 10, 100, empty)
    cv2.createTrackbar("fltr_wht", "TrackBars", 60, 100, empty)

    # While loop until Trackbars is closed
    while cv2.getWindowProperty("TrackBars", 0) >= 0:

        # Retrieves trackbar values
        mode = cv2.getTrackbarPos("mode", "TrackBars")
        kernel_size = cv2.getTrackbarPos("krnl_size", "TrackBars")
        kernel_size_mod = kernel_size % 2  # Ensures that even numbers aren't chosen for gaussian blur
        if kernel_size_mod == 0:
            kernel_size += 1
        sigma = cv2.getTrackbarPos("sigma", "TrackBars")
        sigma = (sigma+1)/10  # Adds 1 since sigma can't be 0 and divides by 10 to get 0.1 intervals
        filter_weight = cv2.getTrackbarPos("fltr_wht", "TrackBars") - 50  # Subtracts 50 so 0 is in the center

        # Creates kernel of given size
        kernel = np.zeros((kernel_size, kernel_size))
        # Center is rounded down to get correct index
        center = int(kernel_size / 2)

        for x in range(kernel_size):
            for y in range(kernel_size):
                # Calculates dist of current element from center of kernel
                center_dist = np.sqrt((x - center)**2 + (y - center)**2)
                # Calculates value of the distance based on gaussian function
                g_x = np.exp(-(center_dist**2/(2*sigma**2))) / (sigma*np.sqrt(2 * np.pi))
                # Sets given element to the gaussian value
                kernel[x, y] = g_x

        # Divides kernel by sum to normalise (kernel now sums to 1)
        kernel /= np.sum(kernel)
        # Applies kernel to image
        result_img = cv2.filter2D(img, -1, kernel)

        # Histogram equalisation
        if mode == 0:
            # Converts to HSV
            img_HSV = cv2.cvtColor(result_img, cv2.COLOR_BGR2HSV)
            # Extracts value channel
            V = img_HSV[:, :, 2]
            # Extracts histogram
            hist, bins = np.histogram(V.flatten(), 256, [0, 256])
            # Calculates cumulative distribution
            cdf = hist.cumsum()
            # Isolates non-zero elements
            cdf_masked = np.ma.masked_equal(cdf, 0)
            # Performs histogram equalisation
            cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
            # Refills masked elements
            cdf = np.ma.filled(cdf_masked, 0).astype(np.uint8)
            # Applies histogram equalisation to channel
            V = cv2.LUT(V.astype(np.uint8), cdf)
            # Recombines channels
            img_HSV[:, :, 2] = V
            result_img = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2BGR)

        # Warth curve
        if mode == 1:
            # Initialises 3 LUT for each channel
            b_LUT, g_LUT, r_LUT = np.arange(256), np.arange(256), np.arange(256)

            # Applies colour curve
            b_LUT -= filter_weight
            r_LUT += filter_weight

            # Stacks each LUT for each channel
            l_table = np.dstack((b_LUT, g_LUT, r_LUT))
            # Clips values to ensure they are in valid range
            l_table = np.clip(l_table, 0, 255)

            # Applies LUT to img
            result_img = cv2.LUT(result_img.astype(np.uint8), l_table)

        # Custom contrast enhancement
        if mode == 2:
            # Reduces filter weight to keep contrast curve correct
            filter_weight = filter_weight/2

            # Generates sample points
            x = np.array([0, 64, 128, 192, 255]).astype(np.float32)
            y = np.array([0, 64 - filter_weight, 128, 192 + filter_weight, 255]).astype(np.float32)

            # Generates line of best fit
            spl = UnivariateSpline(x, y)

            # Applies colour curve to img
            result_img[:, :, 0] = np.clip(spl(result_img[:, :, 0]), 0, 255)
            result_img[:, :, 1] = np.clip(spl(result_img[:, :, 1]), 0, 255)
            result_img[:, :, 2] = np.clip(spl(result_img[:, :, 2]), 0, 255)

            # # Plots colour curve
            # plt.plot(x, y, "ro")
            # plt.plot(np.linspace(0, 255, 255), spl(np.linspace(0, 255, 255)))
            # plt.waitforbuttonpress(1)
            # plt.clf()

        # Shows result
        cv2.imshow("result_img", result_img.astype(np.uint8))
        cv2.waitKey(1)

        # Checks for 'p' press
        if keyboard.is_pressed('p'):
            cv2.imwrite("problem3.jpg", result_img.astype((np.uint8)))
    cv2.destroyAllWindows()

# python ImageProcessing.py problem4 input1.jpg
def problem4(image):
    # Initialises image data
    img = cv2.imread(image).astype(np.float32)
    previous_result_img = np.copy(img)
    center_x = int(img.shape[0] / 2)
    center_y = int(img.shape[1] / 2)

    # Called to by trackbars when they are changed
    def empty(x):
        pass

    # Initialises gaussian filter
    global gaus_filter
    gaus_filter = 1

    # Creates gaussian filter
    def gaus(sigma):
        # Creates filter of given size
        filter = np.zeros_like(img)[:, :, 0]
        sigma += 1  # Can't equal 0

        # Iterates through kernel to generate values
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                # Calculates dist of current element from center of kernel
                center_dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                # Calculates value of the distance based on gaussian function
                g_x = np.exp(-(center_dist ** 2 / (2 * sigma ** 2))) / (sigma * np.sqrt(2 * np.pi))
                # Sets given element to the gaussian value
                filter[x, y] = g_x

        # Normalises to get values between 0 and 1
        filter = filter * 1 / np.amax(filter)

        # Assigns filter to variable
        global gaus_filter
        gaus_filter = filter

    # Initialises gaus filter
    gaus(100)

    # Performs fourier transform for low_pass filter
    def fourier(f_channel, mask):
        f_channel = np.fft.fft2(f_channel)
        f_channel = np.fft.fftshift(f_channel)
        f_channel = f_channel * mask
        f_channel = np.fft.ifftshift(f_channel)
        f_channel = np.fft.ifft2(f_channel)
        f_channel = np.abs(f_channel)
        f_channel = np.clip(f_channel, 0, 255)
        return f_channel

    # Performs bilinear interpolation
    def bilinear(input_x, input_y, img_temp):
        # Calculates 4 points surrounding input_x and y
        x1, x2 = int(np.floor(input_x)), int(np.ceil(input_x))
        y1, y2 = int(np.floor(input_y)), int(np.ceil(input_y))

        if x1 == x2 and y1 == y2:  # If coordinates are already integers
            f_x_y = img_temp[x1, y1]
        elif x1 == x2:  # If x coordinate is integer, performs interpolation in y only
            f_x_y = (y2 - input_y) * img_temp[x1, y1] + (input_y - y1) * img_temp[x1, y2]
        elif y1 == y2:  # If y coordinate is integer, performs interpolation in x only
            f_x_y = (x2 - input_x) * img_temp[x1, y1] + (input_x - x1) * img_temp[x2, y1]
        else:  # Otherwise performs full bilinear interpolation
            f_x_y1 = (x2 - input_x) * img_temp[x1, y1] + (input_x - x1) * img_temp[x2, y1]
            f_x_y2 = (x2 - input_x) * img_temp[x1, y2] + (input_x - x1) * img_temp[x2, y2]
            f_x_y = (y2 - input_y) * f_x_y1 + (input_y - y1) * f_x_y2
        return f_x_y

    # Initialize trackbar data
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("intrplatn", "TrackBars", 0, 1, empty)
    cv2.createTrackbar("low_pass", "TrackBars", 0, 2, empty)
    cv2.createTrackbar("inverse", "TrackBars", 0, 1, empty)
    min_radius = min(center_x, center_y)   # Ensures that swirl radius always lies within the img
    quarter_radius = int(min_radius/4)
    cv2.createTrackbar("swrl_rdus", "TrackBars", quarter_radius, min_radius, empty)
    cv2.createTrackbar("swrl_angle", "TrackBars", 125, 200, empty)
    cv2.createTrackbar("fltr_rdus", "TrackBars", quarter_radius, min_radius, gaus)

    # While loop until Trackbars is closed
    while cv2.getWindowProperty("TrackBars", 0) >= 0:

        # Retrieves trackbar values
        interpolation = cv2.getTrackbarPos("intrplatn", "TrackBars")
        low_pass = cv2.getTrackbarPos("low_pass", "TrackBars")
        inverse = cv2.getTrackbarPos("inverse", "TrackBars")
        swirl_radius = cv2.getTrackbarPos("swrl_rdus", "TrackBars")-1
        swirl_angle = cv2.getTrackbarPos("swrl_angle", "TrackBars")
        swirl_angle = (swirl_angle - 100)/100
        filter_radius = cv2.getTrackbarPos("fltr_rdus", "TrackBars")

        # Performs low pass filter
        if low_pass:

            # Circle filter
            if low_pass == 1:
                mask = np.zeros_like(img)
                mask = cv2.circle(mask, (center_y, center_x), filter_radius, (1, 0, 0), -1)
                mask = mask[:, :, 0]

            # Gaussian mask
            if low_pass == 2:
                mask = gaus_filter

            # Extracts colour channels
            b, g, r = cv2.split(img)
            # Performs low_pass filter
            f_b, f_g, f_r = fourier(b, mask), fourier(g, mask), fourier(r, mask)
            # Recombines to create img
            f_img = cv2.merge((f_b, f_g, f_r)).astype(np.float32)

        else:
            # If no low pass filter then image is unchanged
            f_img = img.copy()

        # Initialises temporary img
        img_temp = np.copy(f_img)
        # Copies img so that any pixels that arent changed by transformation are the same
        result_img = np.copy(f_img)
        # Initialises inverse image to the previous result image
        inverse_img = np.copy(previous_result_img)

        # Iterates through part of img that is affected
        for i in range(center_x-swirl_radius, center_x+swirl_radius):
            for j in range(center_y-swirl_radius, center_y+swirl_radius):

                # Calculates distance of current pixel from center of img
                x_coord = i - center_x
                y_coord = j - center_y

                # Calculates polar coordinates for pixel
                theta = np.arctan2(y_coord, x_coord)
                r = np.sqrt(x_coord ** 2 + y_coord ** 2)

                # Calculates the amount of swirl based on its distance from the center
                swirl_amount = 1 - r/swirl_radius
                # If the pixel is not within the swirl radius it is ignored
                if swirl_amount < 0:
                    continue

                # Adds the swirl angle to the pixel based on how far it is from the center
                theta += swirl_amount * swirl_angle * np.pi * 2

                # Calculates the input coordinates for the output pixel
                input_x = (r * np.cos(theta)) + center_x
                input_y = (r * np.sin(theta)) + center_y

                # Nearest neighbour interpolation
                if interpolation == 0:
                    # Round to nearest neighbouring pixel
                    x = round(input_x)
                    y = round(input_y)
                    result_img[i, j] = img_temp[x, y]

                # Bilinear interpolation
                if interpolation == 1:
                    result_img[i, j] = bilinear(input_x, input_y, img_temp)

                if inverse:
                    theta -= 2*(swirl_amount * swirl_angle * np.pi * 2)
                    inverse_input_x = (r * np.cos(theta)) + center_x
                    inverse_input_y = (r * np.sin(theta)) + center_y

                    if interpolation == 0:
                        inverse_x = round(inverse_input_x)
                        inverse_y = round(inverse_input_y)
                        inverse_img[i, j] = previous_result_img[inverse_x, inverse_y]

                    if interpolation == 1:
                        inverse_img[i, j] = bilinear(inverse_input_x, inverse_input_y, previous_result_img)

        # Shows result
        cv2.imshow("result_img", result_img.astype(np.uint8))

        # Checks for 'p' press
        if keyboard.is_pressed('p'):
            cv2.imwrite("problem4.jpg", result_img.astype((np.uint8)))

        if inverse:
            # Makes previous result_img the current result_img
            previous_result_img = result_img
            img_difference = cv2.subtract(inverse_img, img)
            img_difference = np.clip(img_difference, 0, 255)

            # Shows results
            cv2.imshow("inverse_img", inverse_img.astype(np.uint8))
            cv2.imshow("img_difference", img_difference.astype(np.uint8))

            # Checks for 'p' press
            if keyboard.is_pressed('p'):
                cv2.imwrite("inverse_img.jpg", inverse_img.astype((np.uint8)))
                cv2.imwrite("img_difference.jpg", img_difference.astype((np.uint8)))
        else:
            cv2.destroyWindow("inverse_img")
            cv2.destroyWindow("img_difference")

        cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    globals()[sys.argv[1]](sys.argv[2])