import cv2
import numpy as np

def _build_single_panel(dimensions, value):
    ret = np.ones((dimensions, dimensions,3), dtype=np.uint8) * 255

    if value > 0:
        value = str(value)
    else:
        return ret

    font = cv2.FONT_HERSHEY_PLAIN
    fontscale = 3.5
    color = (0,0,0)
    thickness = 2
    linetype = cv2.LINE_AA
    height, width = cv2.getTextSize(value, font, fontscale, thickness)[0]
    bottom_corner = (dimensions//2 - width//2, dimensions//2 + height//2)
    cv2.putText(ret, value, bottom_corner, font, fontscale, color, thickness, linetype)

    return ret

def display_board(board_state, wait = False, size_px = 640):
    #board state should be passed in as a list of integers or np array of shape 9x9

    panel_size = size_px // 9
    internal_border_size = 2
    ninth_extra_width = 4
    external_border_size = 5
    total_size = (8*internal_border_size) + (2*external_border_size) + (2*ninth_extra_width) + 2 + (9 * panel_size)


    board = np.zeros((total_size, total_size,3), dtype=np.uint8)

    x = 0
    y = 0
    for value in board_state:
        panel = _build_single_panel(panel_size, value)

        col1 = (x*panel_size) + external_border_size
        if x > 0:
            col1 += (x) * internal_border_size
            col1 += x//3 * ninth_extra_width


        row1 = (y * panel_size) + external_border_size
        if y > 0:
            row1 += (y) * internal_border_size
            row1 += y//3 * ninth_extra_width

        board[row1:row1+panel_size, col1:col1+panel_size] = panel
        x+=1
        if x > 8:
            x = 0
            y+=1

    cv2.imshow('Sudoku Solver', board)
    if wait:
        cv2.waitKey()
    else:
        cv2.waitKey(1)
