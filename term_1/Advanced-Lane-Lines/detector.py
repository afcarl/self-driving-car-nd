import cv2
import line
import math
import processing as proc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Detector():
    def __init__(self):
        self.line_left = line.Line()
        self.line_right = line.Line()
        self.detected = False
        self.iteration = 0
        self.lane_widths = [] # save lane width for last 10 iteration 
        self.lane_width = None
        self.algorithms = [proc.gradient_color, proc.yellow_white, proc.gradient_saturation]  
        self.current_option = 0

    def detect_lane(self, image, debug=False):
        """ find and update lane searching result on each video frame
            This is done by iterate through different image algorithm to detect line in different conditions
        """
        try_count = 0
        self.current_option = 0
        while try_count < len(self.algorithms):
            
            binary_warped = self.algorithms[self.current_option](image)
            # Take a histogram of the whole image
            histogram = np.sum(binary_warped, axis=0)

            # Find the peak of the left and right halves of the histogram
            midpoint = np.int(histogram.shape[0]/2)

            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
            
            # find lane index starting with base position 
            left_line_position, left_windows = self.find_line(leftx_base, binary_warped, self.line_left.current_fit)
            right_line_position, right_windows = self.find_line(rightx_base,binary_warped, self.line_right.current_fit)
            
            if (not left_line_position) or (not right_line_position):
                self.detected = False
                return

            left_line = self.get_line(left_line_position, binary_warped.shape[0])  
            right_line = self.get_line(right_line_position, binary_warped.shape[0])
            
            if self.is_good_candidate(left_line, right_line):
                self.detected = True
                self.line_left = left_line
                self.line_right = right_line
            else:
                self.detected = False
                # switch to another image processing algorithm
                self.current_option = (self.current_option + 1) % len(self.algorithms) 
            
            if debug:
                self.show_debug(binary_warped, left_line, right_line, left_windows, right_windows)
            
            if self.detected:
                break 

            try_count += 1

    def is_good_candidate(self, left, right):
        margin = 100
        leftx = np.mean(left.fitx)
        rightx = np.mean(right.fitx)
        left_theta = self.theta(left.fitx)
        right_theta = self.theta(right.fitx)
        # To estimate lane width, assuming lane width does not change over the video clip, take initial 5 frames
        if self.lane_width == None:
            if self.iteration < 5: 
                self.lane_widths.append(rightx - leftx)
                self.iteration += 1
            else:
                self.lane_width = np.mean(self.lane_widths)   
            return False
        
        jitter_threshold = 100
        is_x_jump = abs(leftx - np.mean(self.line_left.fitx)) > jitter_threshold or \
                  abs(rightx - np.mean(self.line_right.fitx)) > jitter_threshold
        is_correct_distance = abs((rightx - leftx) - self.lane_width) <= margin
        theta_threshold = 25
        is_same_sign =  (left_theta * right_theta > 0)
        is_parallel = (abs(left_theta - right_theta) < theta_threshold)
        #print("parallel ", is_parallel, "is distance ", is_correct_distance, "jmp ", is_x_jump)
        return (not is_x_jump) and is_correct_distance and is_parallel 
    
    def to_point(self, line):
        x1,y1,x2,y2 = line
        if x2-x1 == 0:
            return 0, x1
        if y2-y1 == 0:
            return np.pi/2, y1
        slope = (y2-y1)/(x2-x1)
        theta = math.atan(-1/slope)
        rho = x1*math.cos(theta) + y1*math.sin(theta)
        return theta,rho
  
    def theta(self, fitx):
        y1 = 719
        x1 = fitx[-1]
        y2 = 0
        x2 = fitx[0]
        theta, rho = self.to_point((x1, y1, x2, y2))
        return theta*(180/np.pi)

    def show_debug(self, binary_warped, left_line, right_line, left_windows, right_windows):
        print("lane_width ", self.lane_width, "detected ", self.detected, "next option ", \
                self.current_option, 'left_theta ', left_line.theta, 'right_theta ', right_line.theta)
        out_image = self.draw_windows(binary_warped, left_windows, right_windows)
        out_image[left_line.ally, left_line.allx] = [255, 0, 0]
        out_image[right_line.ally, right_line.allx] = [0, 0, 255]
        ploty = np.linspace(0, binary_warped.shape[0]-1,binary_warped.shape[0])
        fig = plt.figure()
        fig.add_subplot(1,2,1)
        plt.imshow(binary_warped, 'gray')
        fig.add_subplot(1,2,2)
        plt.imshow(out_image)
        self.draw_line_fit(left_line.fitx, right_line.fitx, ploty)
        #histogram = np.sum(binary_warped, axis=0)
        #plt.plot(histogram)
        plt.show()

    def get_line(self, line_pos, image_height):
        l = line.Line()
        l.allx, l.ally = zip(*line_pos)
        l.current_fit = np.polyfit(l.ally, l.allx, 2)
        ploty = np.linspace(0, image_height-1, image_height)
        l.fitx = l.current_fit[0]*ploty**2 + l.current_fit[1]*ploty + l.current_fit[2]
        l.curverad = self.calculate_curvature(l, ploty)
        l.theta = self.theta(l.fitx)
        return l

    def find_line(self, current_x, binary_warped, fit, margin=60, minpix=50, nwindows=9):
        """ find line pixel position
            return a list of pixel position (x,y) in the wapred_binary image and sliding windows
        """
        image_height = binary_warped.shape[0]
        window_height = np.int(image_height/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzeroy = np.array(binary_warped.nonzero()[0])
        nonzerox = np.array(binary_warped.nonzero()[1])
        
        windows = []
        line_position = [] # list of pixel position (x,y) considered to be part of lane
        if self.detected == True:
            tmp_x = fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2]
            good_inds = ((nonzerox > (tmp_x - margin)) & (nonzerox < (tmp_x + margin))) 
            line_position = list(zip(nonzerox[good_inds].tolist(), nonzeroy[good_inds].tolist()))
        else:
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = image_height - (window+1)*window_height
                win_y_high = image_height - window*window_height
                win_x_low = current_x - margin
                win_x_high = current_x + margin

                # Identify the nonzero pixels in x and y within the window
                good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
                line_position.extend(list(zip(nonzerox[good_inds].tolist(), nonzeroy[good_inds].tolist())))
                windows.append((win_x_low, win_x_high, win_y_low, win_y_high))
                if len(good_inds) > minpix: # minumum pixels to recenter the window
                    current_x = np.int(np.mean(nonzerox[good_inds]))
        return line_position, windows

    def calculate_curvature(self, line, ploty):
        """ calculate curvature
        """
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # Fit new polynomials to x,y in world space
        y_eval = np.max(ploty)
        fit_cr = np.polyfit(ploty * ym_per_pix, line.fitx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        # Now our radius of curvature is in meters
        return curverad

    def calculate_car_position(self, left, right):
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        image_center = 640 # image center
        leftx = np.mean(left.fitx)
        rightx = np.mean(right.fitx)
        lane_center = leftx + (rightx - leftx) / 2
        #print(leftx, rightx, lane_center)
        return xm_per_pix * (image_center - lane_center) # negative means vehicle is to the left of the lane center

    def draw_windows(self, source, left_windows, right_windows):
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((source, source, source))*255
        for i in range(len(left_windows)):
            win_xleft_low, win_xleft_high, win_y_low, win_y_high = left_windows[i]
            win_xright_low, win_xright_high, win_y_low, win_y_high = right_windows[i]
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        return out_img

    def draw_line_fit(self, left_fitx, right_fitx, ploty):
        # Generate x and y values for plotting
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

    def draw_final_lane(self, image):
        """ draw the detected lane lines back on the original image
        """
        if (len(self.line_left.fitx) == 0):
            return image
        # Create an image to draw the lines on
        color_warp = np.zeros_like(image).astype(np.uint8)

        ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.line_left.fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.line_right.fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        Minv = proc.get_perspective_matrix(True)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        text = 'Curvature: left {left_curverad:.1f}, right {right_curverad:.1f}'.format(left_curverad=self.line_left.curverad, \
                                                                                    right_curverad=self.line_right.curverad)
        cv2.putText(result, text, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        car_pos = self.calculate_car_position(self.line_left, self.line_right)
        cv2.putText(result, "Car position: {:4.2f}m".format(car_pos), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return result
               
