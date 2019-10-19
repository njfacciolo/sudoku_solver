from puzzle import Puzzle
import numpy as np
import cv2
import csv
import time


if __name__ == "__main__":
    filepath = './puzzles/sudoku.csv'
    with open(filepath, newline='') as pz:
        reader = csv.reader(pz,)
        next(reader, None) # Skip the header line

        perf = 0
        ct = 0
        try:
            for input, solution in reader:
                s= time.perf_counter()
                ct+=1
                input = np.reshape(np.asarray([int(num) for num in input.strip()]), (9, 9))
                output = Puzzle(input).solve()
                if ''.join(output) != solution:
                    print('Puzzle: {} was solved incorrectly')
                print('Puzzle: {}   Time to solve: {:.3f}'.format((ct), (time.perf_counter() - s)))
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()



