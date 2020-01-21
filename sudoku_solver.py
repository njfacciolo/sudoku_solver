from puzzle import Puzzle
from sudoku_visualizer import Visualizer
from puzzle_image_extractor import extract_puzzle
import numpy as np
import cv2
import csv
import time
import os

def _check_solution(real_solution, solved_puzzle, puzzlenumber, solver):
    puzzle_string = ''.join( str(x) for r in solved_puzzle for x in r)
    if puzzle_string != real_solution:
        pz = Puzzle(np.reshape(np.asarray([int(num) for num in solution.strip()]), (9, 9)))
        pz.display_board(name='Correct Solution')
        print('Puzzle: {} was solved incorrectly by {}'.format(puzzlenumber, solver))
        cv2.waitKey(0)


if __name__ == "__main__":
    show_recursive = True
    show_standard = False
    puzzle_extractor_display_time = 100 # -1 waits for user at each step, 0 displays nothing, else display in ms

    images_path = 'puzzles/puzzle_pictures/'
    image_types = ['.jpg', '.JPG', '.png', '.PNG']
    image_files = [file for file in os.listdir(images_path) for image_type in image_types
                   if file.endswith(image_type)]

    for image_name in image_files:
        image_gray = cv2.imread(images_path + image_name, 0)
        if image_gray is None:
            continue
        puzzle_state, puzzle_image = extract_puzzle(image_gray, puzzle_extractor_display_time)
        new_visualizer = Visualizer(puzzle_state, puzzle_image, window_name='Recursive Solver')
        input = np.reshape(np.asarray([int(num) for num in puzzle_state.strip()]), (9, 9))
        # new_puzzle = Puzzle(input, visualizer=new_visualizer, display_updates=show_standard).solve()
        Puzzle(visualizer=new_visualizer).solve_recursive(input, display_updates=show_recursive)



    filepath = './puzzles/sudoku.csv'
    with open(filepath, newline='') as pz:
        reader = csv.reader(pz,)
        next(reader, None) # Skip the header line

        recursive_sum = 0
        standard_sum = 0

        perf = 0
        ct = 0
        try:
            for input, solution in reader:
                ct+=1

                if show_recursive:
                    Visualizer(([int(x) for x in input]), window_name='Recursive Solver')
                if show_standard:
                    Visualizer(([int(x) for x in input]))

                input = np.reshape(np.asarray([int(num) for num in input.strip()]), (9, 9))

                start_standard = time.perf_counter()
                standard_out = Puzzle(input, display_updates=show_standard).solve()
                standard_time = time.perf_counter() - start_standard

                start_recursive = time.perf_counter()
                recursive_out = Puzzle().solve_recursive(input, display_updates=show_recursive)
                recursive_time = time.perf_counter() - start_recursive

                recursive_sum += recursive_time
                standard_sum += standard_time

                _check_solution(solution, recursive_out, ct, 'Recursive Solver')
                _check_solution(solution, standard_out, ct, 'Standard Solver')

                print('Puzzle: {}   Recursive time: {:.3f}  Standard time: {:.3f}'.format((ct), recursive_time,
                                                                                          standard_time ))
                print('Puzzle: {}   Recursive total: {:.3f}  Standard total: {:.3f}'.format((ct), recursive_sum,
                                                                                          standard_sum))
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()



