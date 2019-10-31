import cv2
import numpy as np

class Visualizer:
    def __init__(self, board_start_state, canvas=None, window_name = 'Solver', default_edge_length = 640):
        '''
        Set up the visualizer for a new board.
        :param board_start_state: String or list of 81 integers representing the state of the board
        :param canvas: image to overlay new information on. If None, will build the default display
        :param default_edge_length: default edge length if creating new display
        '''
        self.canvas = canvas if canvas is not None else self._build_canvas(board_start_state)
        self.start_state = board_start_state
        self.edge_len = canvas.shape[0]
        self.window_name = window_name
        self.internal_border_size = self.edge_len // 300
        self.ninth_extra_width = self.internal_border_size * 2
        self.external_border_size = self.ninth_extra_width + 1
        self.panel_size = (self.edge_len - (8 * self.internal_border_size)
                        - (2 * self.external_border_size)
                        -(2 * self.ninth_extra_width))//9

    def _build_canvas(self, start_state):
        new_canvas = np.zeros((self.edge_len, self.edge_len, 3), dtype=np.uint8)

        x = 0
        y = 0
        for value in start_state:
            panel = np.ones((self.panel_size, self.panel_size,3), dtype=np.uint8) * 255

            col1, row1 = self._get_target_corner(x, y)

            new_canvas[row1:row1 + self.panel_size, col1:col1 + self.panel_size] = panel

            new_canvas = self._fill_panel(new_canvas, x, y, value)

            x += 1
            if x > 8:
                x = 0
                y += 1

    def _get_target_corner(self, x, y):
        if x < 0 or y < 0:
            return 0,0

        ret_x = (x * self.panel_size) + self.external_border_size
        if x > 0:
            ret_x += (x) * self.internal_border_size
            ret_x += x // 3 * self.ninth_extra_width

        ret_y = (y * self.panel_size) + self.external_border_size
        if y > 0:
            ret_y += (y) * self.internal_border_size
            ret_y += y // 3 * self.ninth_extra_width

        return ret_x, ret_y

    def _fill_panel(self, image, x, y, value, given_value=False):
        '''
        This draws on the provided input image. Do not provide the original canvas
        :param image: input image to draw on
        :param x: x grid location
        :param y: y grid location
        :param value: value to draw
        :param given_value: represents the values initially in the board. Draws them larger
        :return:
        '''
        if value == 0:
            return image

        col, row = self._get_target_corner(x,y)

        font = cv2.FONT_HERSHEY_PLAIN
        fontscale = self.edge_len/200 if not given_value else self.edge_len/170
        color = (0, 0, 0)
        thickness = 2
        linetype = cv2.LINE_AA
        height, width = cv2.getTextSize(str(value), font, fontscale, thickness)[0]
        bottom_corner = (col + self.panel_size // 2 - width // 2, row + self.panel_size // 2 + height // 2)

        cv2.putText(image, str(value), bottom_corner, font, fontscale, color, thickness, linetype)

        return image

    def generate_frame(self, board_state):
        new_frame = np.copy(self.canvas)

        x, y = 0, 0
        for value, initial in zip(board_state, self.start_state):
            if str(initial) == '0' and str(value) != '0':
                new_frame = self._fill_panel(new_frame, x, y, value)
            x += 1
            if x > 8:
                x = 0
                y += 1
        return new_frame

    def display_frame(self, board_state, display_time = 100):
        self.tempframe = self.generate_frame(board_state)
        cv2.imshow(str(self.window_name), self.tempframe)
        cv2.waitKey(display_time)
