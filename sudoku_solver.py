from puzzle import Puzzle
import numpy as np
import cv2

# Puzzle( np.arange(81).reshape((9,9)))
filepath = './puzzles/puzzles.txt'
with open(filepath) as pz:
    input = np.reshape(np.asarray([int(num) for num in pz.readline().strip()]), (9,9))
    puzzle = Puzzle(input)
    puzzle.solve()

    print('Complete')

    cv2.waitKey()
    cv2.destroyAllWindows()



