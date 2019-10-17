import numpy as np
import math

class Puzzle:
    def __init__(self, values = None):
        if values is None:
            inputs = np.zeros((9,9))
        else:
            inputs = np.asarray(values)

        self.nines_list = []
        self.nines = []
        for x in range(3):
            for y in range(3):
                self.nines_list.append( Nine((x,y), inputs[y*3:(y+1)*3, x*3:(x+1)*3]))

        for i in range(0, len(self.nines_list), 3):
            self.nines.append([self.nines_list[i],self.nines_list[i+1],self.nines_list[i+2]])

    def solve(self):
        while True:
            if( self._solved() ): return

            for nine in self.nines_list:
                row = self._get_row(nine.point[1])
                col = self._get_col(nine.point[0])

                knownsingles = nine.get_known()

                for knownsingle in knownsingles:
                    for n in row:
                        n.clear_row(knownsingle.value[0], knownsingle.point[1])
                    for c in col:
                        c.clear_col(knownsingle.value[0], knownsingle.point[0])

                nine.internal_nine_check()

    def _solved(self):
        for nine in self.nines_list:
            if not nine.known:
                return False
        return True

    def _get_row(self, row):
        return [self.nines[i][row] for i in range(3)]

    def _get_col(self, col):
        return [self.nines[col][i] for i in range(3)]

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
        # build list of list of Single objects
            # col then row
        values = np.reshape(values.T, 9)
        self.singles_list = []
        self.singles = []

        for idx in range(len(values)):
            p = (idx%3, idx//3)
            self.singles_list.append(Single(p, values[idx]))

        for i in range(0, len(values), 3):
            self.singles.append([self.singles_list[i],self.singles_list[i+1],self.singles_list[i+2]])

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
        '''
        Initialize this box in the 9x9 grid.
        :param value: known value. Default to 0 if unknown or blank.
        '''
        self.point = point

        self.known = False if value == 0 else True

        if self.known:
            self.value = []
            self.value.append(value)
        else:
            self.value = [x for x in range(1, 10)]

        print(self.value)

    def set_value(self, value):
        self.value = [].append(value)
        self.known = True

    def remove(self, value):
        if self.known: return

        if value in self.value:
            self.value.remove(value)
            self._update()

    def _update(self):
        if self.known == True: return
        if len(self.value) == 1:
            self.known = True


