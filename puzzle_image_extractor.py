import cv2
import numpy as np
import math



class Image_Displayer:
    DISPLAY_TIME = 100
    DISPLAY_SIZE = (960, 1280)
    DISPLAY_NAME = 'Extraction Status'

    @staticmethod
    def set_display_time(display_time):
        Image_Displayer.DISPLAY_TIME = display_time

    @staticmethod
    def display_image(img):
        # If display time is 0, don't show the state
        if Image_Displayer.DISPLAY_TIME == 0:
            return

        cv2.imshow(Image_Displayer.DISPLAY_NAME, Image_Displayer._resize(img))

        # If the display time is less than zero, display until the user triggers a continuation
        if Image_Displayer.DISPLAY_TIME < 0:
            cv2.waitKey()
            return

        # Just display for the set amount of time
        cv2.waitKey(Image_Displayer.DISPLAY_TIME)

    @staticmethod
    def _resize(img):
        rows,cols =img.shape[:2]
        if rows > Image_Displayer.DISPLAY_SIZE[0] or cols > Image_Displayer.DISPLAY_SIZE[1]:
            row_ratio = Image_Displayer.DISPLAY_SIZE[0] / rows
            col_ratio = Image_Displayer.DISPLAY_SIZE[1] / cols

            ratio = row_ratio if row_ratio < col_ratio else col_ratio

            new_rows, new_cols = int(rows * ratio), int(cols * ratio)

            return cv2.resize(img, (new_rows,new_cols))

        return img



def _show_image(img):
    Image_Displayer.display_image(img)

def _safe_convert_to_gray(img):
    ret = img
    if len(img.shape) > 2:
        ret = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _show_image(ret)
    return ret


def _safe_convert_to_color(img, display = False):
    if len(img.shape) < 3:
        ret = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        ret = img
    if display: _show_image(ret)  # Normally don't need to display this
    return ret


def _blur(img, kernel_size):
    ret = cv2.GaussianBlur(img, (kernel_size, kernel_size), 1)
    _show_image(ret)
    return ret


def _adaptive_threshold(img, size, offset):
    # Invert the image since cv contour finder looks for white shapes on black background
    # Black boarders need to be switched to white
    ret = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, size, offset)
    _show_image(ret)
    return ret


def _apply_morphology(img):
    px = img.shape[0] if img.shape[0] < img.shape[1] else img.shape[1]
    kernel_size = int(math.ceil(px/500))
    if kernel_size < 3:
        kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))

    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    _show_image(opened)

    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    _show_image(closed)

    return closed

def _get_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    to_display = cv2.drawContours(_safe_convert_to_color(img), contours, -1, (0, 255, 0), 2)
    _show_image(to_display)
    return contours

def _filter_contours(img, contours):
    rows,cols = img.shape[:2]
    pixels = rows*cols
    min_area = pixels/8

    if len(contours) < 1:
        return []

    largest_contour = sorted(contours, key = cv2.contourArea, reverse=True)[0]
    if cv2.contourArea(largest_contour) < min_area:
        return False

    epsilon = rows/50 if rows < cols else cols/50


    aproximated = cv2.approxPolyDP(largest_contour, epsilon, closed=True)
    to_display = cv2.drawContours(_safe_convert_to_color(img), largest_contour, -1, (0, 255, 0), 3)

    # aproximated[:,0,:] returns an n row by 2 col array of points
    # first index changes the row (set of x, y pair)
    # last index selects column, either x or y

    aprox_len = len(aproximated[:])
    for i in range(aprox_len):
        pt_row, pt_col = aproximated[i][0][0], aproximated[i][0][1] # What is this crap...
        cv2.circle(to_display, (pt_row, pt_col), 3, (0,0,255), -1)

    _show_image(to_display)

    return largest_contour



def extract_puzzle(img, display_step_time=200):
    '''
    Extract a puzzle from a provided image
    :param img: input image. Should be np array of np.uint8
    :param display_step_time: length of time to display each step. -1 will display the image until the user presses next,
    0 will not display the image at all
    :return: Returns a string of integers representing the current puzzle. Zeros represent unknown boxes
    '''

    Image_Displayer.DISPLAY_TIME = display_step_time
    gaussian_kernel_size = 5
    adaptive_kernel_size = 39
    adaptive_offset = 9

    # Puzzle status where 0 represents unknown
    ret = np.zeros((9, 9), dtype=np.uint8)

    gray = _safe_convert_to_gray(img)

    blur = _blur(gray, gaussian_kernel_size)

    binary = _adaptive_threshold(blur, adaptive_kernel_size, adaptive_offset)

    morphology = _apply_morphology(binary)

    contours = _get_contours(morphology)

    puzzle_region = _filter_contours(morphology, contours)




    # Contour Filtering  - find 4 sided figures that take up a majority of the frame?

    # cv2.destroyAllWindows()


# Test
image = cv2.imread('puzzles/test_puzzle_1.jpg')
extract_puzzle(image, -1)