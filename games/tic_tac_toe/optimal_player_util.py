# taken from https://mblogscode.wordpress.com/2016/06/03/python-naughts-crossestic-tac-toe-coding-unbeatable-ai/
from muzero.classes import Action


def testForkMove(b, mark, i):
    # Determines if a move opens up a fork
    bCopy = list(b)
    bCopy[i] = mark
    winningMoves = 0
    for j in range(0, 9):
        if testWinMove(bCopy, mark, j) and bCopy[j] == ' ':
            winningMoves += 1
    return winningMoves >= 2


def checkWin(b, m):
    return ((b[0] == m and b[1] == m and b[2] == m) or  # H top
            (b[3] == m and b[4] == m and b[5] == m) or  # H mid
            (b[6] == m and b[7] == m and b[8] == m) or  # H bot
            (b[0] == m and b[3] == m and b[6] == m) or  # V left
            (b[1] == m and b[4] == m and b[7] == m) or  # V centre
            (b[2] == m and b[5] == m and b[8] == m) or  # V right
            (b[0] == m and b[4] == m and b[8] == m) or  # LR diag
            (b[2] == m and b[4] == m and b[6] == m))  # RL diag


def testWinMove(b, mark, i):
    # b = the board
    # mark = 0 or X
    # i = the square to check if makes a win
    bCopy = list(b)
    bCopy[i] = mark
    return checkWin(bCopy, mark)


def getComputerMove(b, me, rival):
    # Check computer win moves
    for i in range(0, 9):
        if b[i] == 0 and testWinMove(b, me, i):
            return i
    # Check player win moves
    for i in range(0, 9):
        if b[i] == 0 and testWinMove(b, rival, i):
            return i
    # Check computer fork opportunities
    for i in range(0, 9):
        if b[i] == 0 and testForkMove(b, me, i):
            return i
    #  Check player fork opportunities
    for i in range(0, 9):
        if b[i] == 0 and testForkMove(b, rival, i):
            return i
    # Play center
    if b[4] == 0:
        return 4
    # Play a corner
    for i in [0, 2, 6, 8]:
        if b[i] == 0:
            return i
    #Play a side
    for i in [1, 3, 5, 7]:
        if b[i] == 0:
            return i


def get_optimal_action(board, me, rival):
    return Action(getComputerMove(board, me, rival))
