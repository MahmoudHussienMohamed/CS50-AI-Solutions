"""
Tic Tac Toe Player
"""
import copy

X = "X"
O = "O"
EMPTY = None

def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_cnt, o_cnt = 0, 0
    for row in board:
        for cell in row:
            if cell == X:
                x_cnt += 1
            elif cell == O:
                o_cnt += 1
    return X if x_cnt == o_cnt else O

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    empty_cells = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                empty_cells.add((i, j))
    return empty_cells

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    if 0 > i > 2 or 0 > i > 2:
        raise ValueError
    else:
        resultant = copy.deepcopy(board)
        resultant[i][j] = player(board)
        return resultant

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    check_row = lambda row : board[row][0] if board[row][0] != EMPTY and board[row][0] == board[row][1] == board[row][2] else None
    check_col = lambda col : board[0][col] if board[0][col] != EMPTY and board[0][col] == board[1][col] == board[2][col] else None
    for i in range(3):
        w = (check_row(i), check_col(i))
        for j in w:
            if j != None:
                return j
    if board[0][0] != EMPTY and board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    if board[1][1] != EMPTY and board[1][1] == board[0][2] == board[2][0]:
        return board[1][1]
    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) != None:
        return True
    for row in board:
        for cell in row:
            if cell == EMPTY:
                return False
    return True

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    win = winner(board)
    if win == X:
        return 1
    else:
        return -1 if win == O else 0

def min_value(state):
    if terminal(state):
        return utility(state)
    v = 10
    for action in actions(state):
        v = min(v, max_value(result(state, action)))
        if v == -1:
            break
    return v

def max_value(state):
    if terminal(state):
        return utility(state)
    v = -10
    for action in actions(state):
        v = max(v, min_value(result(state, action)))
        if v == 1:
            break
    return v

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    optimal_action = None
    if player(board) == X:
        max_score = -10
        for action in actions(board):
            tmp = max_value(result(board, action))
            if tmp > max_score:
                max_score = tmp
                optimal_action = action
            if max_score == 1:
                break
    else:
        min_score = 10
        for action in actions(board):
            tmp = min_value(result(board, action))
            if tmp < min_score:
                min_score = tmp
                optimal_action = action
            if min_score == -1:
                break
    return optimal_action