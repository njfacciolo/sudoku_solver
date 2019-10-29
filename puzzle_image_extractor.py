import cv2
import numpy as np
import math
import os
from keras.models import model_from_json
# import digit_classifier

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
        rows, cols = img.shape[:2]
        if rows > Image_Displayer.DISPLAY_SIZE[0] or cols > Image_Displayer.DISPLAY_SIZE[1]:
            row_ratio = Image_Displayer.DISPLAY_SIZE[0] / rows
            col_ratio = Image_Displayer.DISPLAY_SIZE[1] / cols

            ratio = row_ratio if row_ratio < col_ratio else col_ratio

            new_rows, new_cols = int(rows * ratio), int(cols * ratio)

            return cv2.resize(img, (new_cols, new_rows))

        return img


def _show_image(img):
    Image_Displayer.display_image(img)


def _load_digit_model():
    model_path = os.path.abspath('digit_model/digit_model.json')
    weights_path = os.path.abspath('digit_model/model.h5')

    # Check if model exists
    if (not os.path.exists(model_path) or not os.path.exists(weights_path)):
        print('Failed to find the digit recognition model.')
        _train_digit_model()
        # return None

    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    print("Successfully Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return loaded_model


def _train_digit_model():
    pass


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


def _normalize_dimensions(img, normalize_px, display = False):
    orig_rows, orig_cols = img.shape[:2]
    row_ratio, col_ratio = normalize_px / orig_rows, normalize_px / orig_cols

    ratio = max(row_ratio, col_ratio)
    # one or more dimension is greater than target, must reduce image size
    if row_ratio < 1 or col_ratio < 1:
        ratio = min(row_ratio, col_ratio)

    return cv2.resize(img, (int(orig_cols * ratio), int(orig_rows * ratio)))


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
    rows, cols = img.shape[:2]
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

    aprox_len = len(aproximated[:])
    ret = []
    for i in range(aprox_len):
        pt_row, pt_col = aproximated[i][0][0], aproximated[i][0][1] # What is this crap...
        ret.append((pt_row, pt_col))
        cv2.circle(to_display, (pt_row, pt_col), 3, (0,0,255), -1)

    _show_image(to_display)

    return ret


def _order_points(pts):
    # Originally from: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    # Light cleanup
    ret = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    ret[0] = pts[np.argmin(s)]
    ret[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    ret[1] = pts[np.argmin(diff)]
    ret[3] = pts[np.argmax(diff)]

    return np.asarray(list(ret))


def _four_point_transform(image, pts):
    # Originally from: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    pts = np.asarray([np.array(pt) for pt in pts], dtype = np.float32)
    pts = _order_points(pts)

    new_width = min(Image_Displayer.DISPLAY_SIZE)
    new_height = new_width

    dst = np.array([
        [0, 0],
        [new_width - 1, 0],
        [new_width - 1, new_height - 1],
        [0, new_height - 1]], dtype=np.float32)

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (new_width, new_height))

    # return the warped image
    _show_image(warped)
    return warped


def _suppress_grid_lines(binary, gray):
    edge_len = binary.shape[0]

    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Debugging aide
    # cv2.imshow('canny', edges)
    # cv2.waitKey(0)
    # cv2.destroyWindow('canny')

    confidence = edge_len//5        # higher value == more confident in line
    rho = 1 # Distance resolution of the accumulator in pixels
    theta = np.pi / 180 # Angle resolution of the accumulator in radians.

    lines = cv2.HoughLines(edges, rho, theta, confidence)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]

            # only draw lines unless they are vertical or horizontal
            deg = theta* 180/np.pi
            for i in range(4):
                if abs(deg - (i*90)) < 7.5:
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                    cv2.line(gray, pt1, pt2, (0), 7, cv2.LINE_AA)     # Use thick lines to cover up the grid
                    cv2.line(binary, pt1, pt2, (0), 7, cv2.LINE_AA)
                    break

    return  gray, binary


def _edge_suppression(img, debug_mode = False):
    min_area = 0.015
    fitment = 1.75
    max_central_offset = 0.2
    orig_rows, orig_cols = img.shape[:2]
    ret = np.zeros((orig_rows,orig_cols), dtype=np.uint8)

    # Invert image since mnist is black background, white digits
    retval, temp = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Get contours
    contours, hierarchy = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check that atleast one contour was found
    if len(contours) < 1:
        return ret, False

    epsilon = 1
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in sorted_contours:
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(polygon)
        center_x, center_y = x + (w // 2), y + (h // 2)

        # Ensure area requirements are met
        if w*h < (orig_rows *orig_cols)*min_area:
            continue

        if abs(center_x - (orig_cols//2)) > int(orig_cols*max_central_offset):
            continue
        if abs(center_y - (orig_rows//2)) > int(orig_rows*max_central_offset):
            return ret, False

        n_h = int(max(w,h)*fitment)

        # Build a new 'region' with the digit in the middle to hopefully improve detection with NN
        new_region = np.zeros((n_h, n_h), dtype=np.uint8)

        row_paste_origin = (n_h//2) - (h//2)
        col_paste_origin  = (n_h//2) - (w//2)

        new_region[row_paste_origin:row_paste_origin+h, col_paste_origin:col_paste_origin+w] = temp[y:y+h, x:x+w]
        new_region = cv2.resize(new_region, (orig_cols,orig_rows))

        return new_region, True

    return ret, False


def _get_values(model, binary, gray):
    # Img should be de-warped and square already. Goal is to now divide it into
    # blocks and perform number extraction

    rows, cols = binary.shape[:2]
    if rows != cols:
        return ''

    step_size = rows/9



    suppressed_grid_gray, suppressed_grid_binary = _suppress_grid_lines(binary, gray)
    # _show_image(suppressed_grid_gray)
    _show_image(suppressed_grid_binary)

    classified_values = np.zeros((9,9,2))

    nn_input_len = 28*28
    for row in range(9):
        for col in range(9):
            slice_row1 = int(row*step_size)
            slice_row2 = int((row+1)*step_size)
            slice_col1 = int(col*step_size)
            slice_col2 = int((col+1)*step_size)
            region = suppressed_grid_binary[slice_row1:slice_row2, slice_col1:slice_col2 ]
            # cv2.imshow('pre-suppression', region)

            region, found_digit = _edge_suppression(region, debug_mode=True)
            suppressed_grid_binary[slice_row1:slice_row2, slice_col1:slice_col2 ] = region

            if not found_digit:
                classified_values[row, col, :] = [0, 1]
                continue

            # compressed = cv2.morphologyEx(region, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
            compressed = cv2.GaussianBlur(region,(3,3), 1)
            compressed = cv2.resize(compressed, (28, 28)).astype('float32')

            # cv2.imshow('post-suppression', compressed)

            inputs = compressed / 255
            inputs = np.reshape(inputs, [1,28, 28, 1]).astype('float32')

            model_prediction = model.predict(inputs)
            digit = np.argmax(model_prediction)
            confidence = model_prediction[0][digit]
            print('Digit: {}   Confidence: {:.3f}'.format(digit, confidence))

            classified_values[row,col,:] = [digit, confidence]


    colored = _safe_convert_to_color(suppressed_grid_binary)
    # Draw grid lines
    for row in range(1, 9):
        cv2.line(colored, (int(row * step_size), 0), (int(row * step_size), cols - 1), (0, 255, 0), 2)
    for col in range(1, 9):
        cv2.line(colored, (0, int(col * step_size)), (rows - 1, int(col * step_size)), (0, 255, 0), 2)
    # _show_image(colored)

    detected_board_string = ''
    for row in range(9):
        for col in range(9):
            origin = (int(col * step_size + 10), int((row * step_size) + 50))
            digit, confidence = classified_values[row, col, :]
            cv2.putText(colored, str(int(digit)), origin, cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            detected_board_string+= str(int(digit))


    _show_image(colored)

    return detected_board_string


def extract_puzzle(img, display_step_time=200):
    '''
    Extract a puzzle from a provided image
    :param img: input image. Should be np array of np.uint8
    :param display_step_time: length of time to display each step. -1 will display
     the image until the user presses next, 0 will not display the image at all
    :return: Returns a string of integers representing the current puzzle. Zeros represent unknown boxes
    '''

    Image_Displayer.DISPLAY_TIME = display_step_time
    normalized_max_dimension = 900 #px
    gaussian_kernel_size = 5
    adaptive_kernel_size = 31
    adaptive_offset = 9

    # ML digit classifier
    model = _load_digit_model()
    if model is None:
        return

    # Print information about each layer's outputs and trainable values
    # print(model.summary())

    # Puzzle status where 0 represents unknown
    ret = ''.join(str(0) for x in range(81))

    gray = _safe_convert_to_gray(img)

    normalized = _normalize_dimensions(gray, normalized_max_dimension)

    blur = _blur(normalized, gaussian_kernel_size)

    binary = _adaptive_threshold(blur, adaptive_kernel_size, adaptive_offset)

    # morphology = _apply_morphology(binary)

    contours = _get_contours(binary)

    puzzle_region = _filter_contours(binary, contours)

    if puzzle_region is None or len(puzzle_region) != 4:
        print('Failed to find any suitable contours representative of a sudoku board.')
        return ret  # Empty board

    transformed_gray = _four_point_transform(normalized, puzzle_region)
    transformed_binary = _four_point_transform(binary, puzzle_region)

    values = _get_values(model, transformed_binary, transformed_gray)

    return values


if __name__ == "__main__":
    display_speed = 150             # -1 waits for user at each step, 0 displays nothing, else display in ms

    image_types = ['.jpg', '.JPG', '.png', '.PNG']
    target_dir = 'puzzles/puzzle_pictures/'
    allfiles = os.listdir(target_dir)
    image_files = [file for file in allfiles for image_type in image_types if file.endswith(image_type)]

    with open(target_dir + "test_answers.txt", "r") as txt_file:
        answers = txt_file.readlines()

    for target, answer_raw in zip(image_files, answers):
        image = cv2.imread(target_dir + target)
        prediction = extract_puzzle(image, display_speed)

        answer = answer_raw.strip()

        total_digits = 0
        total_correct = 0
        incorrect = []
        for value in range(len(answer)):
            if answer[value] != str(0):
                total_digits += 1
                a = answer[value]
                p = prediction[value]
                if answer[value] == prediction[value]:
                    total_correct +=1
                else:
                    incorrect.append(value)

        print('Correctly detected digits: {}/{}, {:.2f}%'.format(total_correct, total_digits,
                                                                 100 * (total_correct / total_digits)))
        if total_digits is not total_correct:
            print('Detected board: {}'.format(prediction))
            print('Expected board: {}'.format(answer))
        cv2.waitKey()

    cv2.destroyAllWindows()



