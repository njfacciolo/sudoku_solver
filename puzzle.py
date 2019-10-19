import numpy as np
import math
from sudoku_visualizer import display_board

class Puzzle:
    def __init__(self, values = None):
        if values is None:
            inputs = np.zeros((9,9))
        else:
            inputs = np.asarray(values)

        self.nines_list = []
        self.nines = []
        for y in range(3):
            for x in range(3):
                ninth = inputs[y*3:(y+1)*3, x*3:(x+1)*3]
                self.nines_list.append( Nine((x,y), ninth))

        for i in range(0, len(self.nines_list), 3):
            self.nines.append([self.nines_list[i],self.nines_list[i+1],self.nines_list[i+2]])

    def solve(self):
        # TODO: bug identified at puzzle 3611, solver gets stuck. Puzzle is solvable. Need 'forward pass' logic.
        # If only one single in any given nine is the only single that has that digit as a remaining possibility,
        # than that single must be that digit, regardless of how many other digits are also 'available'
        display_board(self._get_status())
        while True:
            if( self._solved() ): return self._get_status(characters=True)

            for nine in self.nines_list:

                xval = nine.point[0]
                yval = nine.point[1]
                row = self._get_row(yval)
                col = self._get_col(xval)

                knownsingles = nine.get_known()

                for knownsingle in knownsingles:
                    for n in row:
                        n.clear_row(knownsingle.value[0], knownsingle.point[1])
                    for c in col:
                        c.clear_col(knownsingle.value[0], knownsingle.point[0])

                nine.internal_nine_check()

                display_board(self._get_status())

    def _solved(self):
        for nine in self.nines_list:
            if not nine.known:
                return False
        return True

    def _get_row(self, row):
        ret = []
        for nine in self.nines_list:
            if nine.point[1] == row: ret.append(nine)
        return ret

    def _get_col(self, col):
        ret = []
        for nine in self.nines_list:
            if nine.point[0] == col: ret.append(nine)
        return ret

    def _get_status(self, characters = False):
        '''
        Gather 81 values. Use zeros where unsure
        :return: List of values of each Single
        '''

        ret = np.zeros((9,9), dtype=np.uint8)

        for nine in self.nines_list:
            p = nine.point
            p = (p[1] * 3, p[0]*3)
            values = nine.get_values()

            ret[p[0]:p[0]+3, p[1]:p[1]+3] = values

        ret = ret.tolist()
        rtn = []
        for x in ret:
            rtn = rtn + x

        if not characters:
            return rtn
        return [str(x) for x in rtn]

class Nine:
    def __init__(self, point, values = 0):
        '''
        Initialize this grid of 3x3 within the full 9x9 grid
        :param point: Location of the 3x3 grid within the full grid. Can range from 0-2 for each axis
        :param values: np array of uint size 3x3 indicating the values of the Singles where 0 is unknown
        '''

        self.point = point
        self.known = False

        values = np.asarray(values)

        self.singles_list = []
        self.singles = []

        for y in range(3):
            for x in range(3):
                self.singles_list.append(Single((x,y), values[y,x]))

        for i in range(0, len(self.singles_list), 3):
            self.singles.append([self.singles_list[i],self.singles_list[i+1],self.singles_list[i+2]])
        # print(singles)

    def _update(self):
        for single in self.singles_list:
            if not single.known:
                return
        self.known = True

    def internal_nine_check(self):
        if self.known: return

        known = self.get_known()
        if len(known) < 1: return

        for s in self.singles_list:
            if s.known:
                continue

            for know in known:
                s.remove(know.value[0])

        self._update()

    def get_known(self):
        known = []
        for single in self.singles_list:
            if single.known:
                known.append(single) # value within 3x3 grid

        # if unknown, can we identify only row or column?
        # add more checking

        return known

    def get_values(self):
        ret = np.zeros((3,3), dtype=np.uint8)
        for single in self.singles_list:
            p = single.point
            p = (p[1], p[0])
            if single.known:
                ret[p] = single.value[0]

        return  ret

    def clear_row(self, val, row):
        for single in self.singles_list:        # this scans all nine when we only need to check three
            if single.point[1] == row:
                single.remove(val)

    def clear_col(self, val, col):
        for single in self.singles_list:        # this scans all nine when we only need to check three
            if single.point[0] == col:
                single.remove(val)

class Single:
    def __init__(self, point ,value=0):
        self.point = point

        self.known = False if value == 0 else True

        if self.known:
            self.value = []
            self.value.append(value)
        else:
            self.value = [x for x in range(1, 10)]

    def remove(self, value):
        if self.known: return

        if value in self.value:
            self.value.remove(value)
            self._update()

    def _update(self):
        if self.known == True: return

        if len(self.value) == 1:
            self.known = True


