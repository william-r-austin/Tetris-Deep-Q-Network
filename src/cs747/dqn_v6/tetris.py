"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
from PIL import Image
import cv2
from matplotlib import style
import random

style.use("ggplot")

class Tetris:
    piece_colors = [
        (0, 0, 0),
        (255, 255, 0),
        (147, 88, 254),
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (254, 151, 32),
        (0, 0, 255)
    ]

    pieces = [
        [[1, 1],
         [1, 1]],

        [[0, 2, 0],
         [2, 2, 2]],

        [[0, 3, 3],
         [3, 3, 0]],

        [[4, 4, 0],
         [0, 4, 4]],

        [[5, 5, 5, 5]],

        [[0, 0, 6],
         [6, 6, 6]],

        [[7, 0, 0],
         [7, 7, 7]]
    ]
    
    # If the agent choose to just spin or move the block left/right continuously, we should automatically
    # move it down after a while to prevent infinite loops
    max_sideways_moves = 8

    def __init__(self, height=20, width=10, render_flag=False, block_size=30, gamma=0.99):
        self.height = height
        self.width = width
        self.render_flag = render_flag
        self.gamma = gamma
        
        if render_flag:
            self.block_size = block_size
            self.extra_board = np.ones((self.height * self.block_size, self.width * int(self.block_size / 2), 3), 
                                       dtype=np.uint8) * np.array([204, 204, 255], dtype=np.uint8)
            self.text_color = (200, 20, 220)
        
        self.create_actions()
        
        self.reset()

    def create_actions(self):
        self.actions = {"MOVE_LEFT": lambda: self.moveLeft(), 
                        "MOVE_RIGHT" : lambda: self.moveRight(), 
                        "MOVE_DOWN": lambda: self.moveDown(), 
                        "DROP": lambda: self.drop(), 
                        "ROTATE_CLOCKWISE": lambda: self.rotate_clockwise(), 
                        "ROTATE_COUNTERCLOCKWISE": lambda: self.rotate_counter_clockwise()}
        
        self.action_names = list(self.actions.keys())

    def get_action_names(self):
        return self.action_names

    def reset(self):
        self.board = [[0] * self.width for _ in range(self.height)]
        self.discounted_reward = 0.0
        self.score = 0
        self.tetrominoes = 0
        self.cleared_lines = 0
        self.action_count = 0
        self.sideways_moves_count = 0
        self.bag = list(range(len(self.pieces)))
        random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        self.gameover = False

    def rotate(self, piece):
        num_rows_orig = num_cols_new = len(piece)
        num_rows_new = len(piece[0])
        rotated_array = []

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[j] = piece[(num_rows_orig - 1) - j][i]
            rotated_array.append(new_row)
        return rotated_array

    def rotateCC(self, piece):
        num_rows_orig = num_cols_new = len(piece)
        num_rows_new = len(piece[0])
        rotated_array = []

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[num_cols_new - j - 1] = piece[(num_rows_orig - 1) - j][i]
            rotated_array.append(new_row)
        return rotated_array
   
    def get_current_board_state(self):
        board = [x[:] for x in self.board]
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                board[y + self.current_pos["y"]][x + self.current_pos["x"]] = -1 * self.piece[y][x]
        return board

    def new_piece(self):
        if not len(self.bag):
            self.bag = list(range(len(self.pieces)))
            random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2,
                            "y": 0}
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def check_collision(self, piece, pos):
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                future_y = pos["y"] + y
                future_x = pos["x"] + x 
                boardViolation = future_y > self.height - 1 or future_x < 0 or future_x > self.width - 1
                
                if boardViolation:
                    return True
                else:
                    pieceOverlaps = self.board[future_y][future_x] and piece[y][x]
                    if pieceOverlaps:
                        return True
                    
        return False

    def store(self, piece, pos):
        board = [x[:] for x in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x]
        return board

    def check_cleared_rows(self, board):
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board

    def remove_row(self, board, indices):
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(self.width)]] + board
        return board
    
    def do_action_by_id(self, action_id):
        result_map = None
        if action_id >= 0 and action_id < len(self.action_names):
            action_name = self.action_names[action_id]
            action_function = self.actions[action_name]
            result_map = action_function()
            self.discounted_reward = result_map["reward"] + (self.gamma * self.discounted_reward)
            self.action_count += 1
        else:
            print("Invalid move received.")
            result_map = {}
        
        return result_map
            
    
    def do_action_by_name(self, action_name):
        result_map = None

        if action_name in self.action_names:
            action_function = self.actions[action_name]
            result_map = action_function()
            self.discounted_reward = result_map["reward"] + (self.gamma * self.discounted_reward)
            self.action_count += 1
        else:
            print("Invalid move received.")
            result_map = {}
        
        return result_map
    
    def moveLeft(self):
        next_pos = {"x": self.current_pos["x"] - 1, "y": self.current_pos["y"]}
        return self.executeMove(next_pos, self.piece, False)
    
    def moveRight(self):
        next_pos = {"x": self.current_pos["x"] + 1, "y": self.current_pos["y"]}
        return self.executeMove(next_pos, self.piece, False)
    
    def moveDown(self):
        next_pos = {"x": self.current_pos["x"], "y": self.current_pos["y"] + 1}
        return self.executeMove(next_pos, self.piece, True)
    
    def drop(self):
        move_down_result = self.moveDown()
        while not move_down_result["finalized"]:
            move_down_result = self.moveDown()
        
        return move_down_result
    
    def rotate_clockwise(self):
        next_piece = self.rotate(self.piece)
        return self.executeMove(self.current_pos, next_piece, False)
    
    def rotate_counter_clockwise(self):
        next_piece = self.rotateCC(self.piece)
        return self.executeMove(self.current_pos, next_piece, False)
    
    def finalize_piece(self):
        self.board = self.store(self.piece, self.current_pos)
        lines_cleared, self.board = self.check_cleared_rows(self.board)
        old_reward = 1 + (lines_cleared ** 2) * self.width
        new_reward = 2 + (lines_cleared ** 2) * 15
        
        self.tetrominoes += 1
        self.cleared_lines += lines_cleared
        
        self.new_piece()
        
        # The gameover flag is set by new_piece()
        if self.gameover:
            new_reward -= 20
            old_reward -= 2
        
        self.score += old_reward

        return new_reward, old_reward
    
    """
        Returns a tuple (score, game over, is piece finalized)
    """
    def executeMove(self, next_pos, next_piece, is_down_move):
        move_collision = self.check_collision(next_piece, next_pos)
        result_map = {"reward": -0.05, "gameover": self.gameover, "finalized": False, "score_delta": 0}

        if is_down_move:
            self.sideways_moves_count = 0
            
            if move_collision:
                move_reward, score_delta = self.finalize_piece()
                result_map["reward"] = move_reward
                result_map["gameover"] = self.gameover
                result_map["finalized"] = True
                result_map["score_delta"] = score_delta
            else:
                self.piece = next_piece
                self.current_pos = next_pos
        else:
            self.sideways_moves_count += 1
            
            # If we have a collision OR exceeded the sideways move limit, then
            # just move the block down. Otherwise, move the block to the new location
            if move_collision or self.sideways_moves_count > self.max_sideways_moves:
                result_map = self.moveDown()
            else:
                self.piece = next_piece
                self.current_pos = next_pos
              
        if self.render_flag:
            self.render()
        
        #return reward, self.gameover, is_piece_finalized, score_delta
        return result_map

    def render(self, video=None):
        if not self.gameover:
            img = [self.piece_colors[p] for row in self.get_current_board_state() for p in row]
        else:
            img = [self.piece_colors[p] for row in self.board for p in row]
        img = np.array(img).reshape((self.height, self.width, 3)).astype(np.uint8)
        img = img[..., ::-1]
        img = Image.fromarray(img, "RGB")

        img = img.resize((self.width * self.block_size, self.height * self.block_size))
        img = np.array(img)
        img[[i * self.block_size for i in range(self.height)], :, :] = 0
        img[:, [i * self.block_size for i in range(self.width)], :] = 0

        img = np.concatenate((img, self.extra_board), axis=1)


        cv2.putText(img, "Score:", (self.width * self.block_size + int(self.block_size / 2), self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.score),
                    (self.width * self.block_size + int(self.block_size / 2), 2 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        cv2.putText(img, "Pieces:", (self.width * self.block_size + int(self.block_size / 2), 4 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.tetrominoes),
                    (self.width * self.block_size + int(self.block_size / 2), 5 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        cv2.putText(img, "Lines:", (self.width * self.block_size + int(self.block_size / 2), 7 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.cleared_lines),
                    (self.width * self.block_size + int(self.block_size / 2), 8 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        if video:
            video.write(img)

        cv2.imshow("Deep Q-Learning Tetris", img)
        cv2.waitKey(1)
