from puzzle import Puzzle
from sudoku_visualizer import display_board
import numpy as np
import cv2
import csv
import time

def _check_solution(real_solution, solved_puzzle, puzzlenumber, solver):
    puzzle_string = ''.join( str(x) for r in solved_puzzle for x in r)
    if puzzle_string != real_solution:
        pz = Puzzle(np.reshape(np.asarray([int(num) for num in solution.strip()]), (9, 9)))
        pz.display_board(name='Correct Solution')
        print('Puzzle: {} was solved incorrectly by {}'.format(puzzlenumber, solver))
        cv2.waitKey(0)


if __name__ == "__main__":
    filepath = './puzzles/sudoku.csv'
    with open(filepath, newline='') as pz:
        reader = csv.reader(pz,)
        next(reader, None) # Skip the header line

        show_recursive = False
        show_standard = False

        recursive_sum = 0
        standard_sum = 0

        perf = 0
        ct = 0
        try:
            for input, solution in reader:
                ct+=1
                input = np.reshape(np.asarray([int(num) for num in input.strip()]), (9, 9))

                start_standard = time.perf_counter()
                standard_out = Puzzle(input).solve(display_updates=show_standard)
                standard_time = time.perf_counter() - start_standard

                start_recursive = time.perf_counter()
                recursive_out = Puzzle().solve_recursive(input, display_updates=show_recursive)
                recursive_time = time.perf_counter() - start_recursive

                recursive_sum += recursive_time
                standard_sum += standard_time

                _check_solution(solution, recursive_out, ct, 'Recursive Solver')
                _check_solution(solution, standard_out, ct, 'Standard Solver')

                print('Puzzle: {}   Recursive time: {:.3f}  Standard time: {:.3f}'.format((ct), recursive_time, standard_time ))
                print('Puzzle: {}   Recursive avg: {:.3f}  Standard avg: {:.3f}'.format((ct), recursive_sum/ct,
                                                                                          standard_sum/ct))
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()



