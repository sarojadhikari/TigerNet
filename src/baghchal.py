import numpy as np


def distance_square(x1, y1, x2, y2):
    return (x2 - x1) ** 2.0 + (y2 - y1) ** 2.0


class Baghchal:
    """Baghchal board
    
    define the board setup, pieces, valid moves and other rules
    """

    def __init__(self):
        """Initialize original state of the board
        
        Board has 4 tigers at the four corners and is otherwise empty
        
        The first turn is that of the goat -- the goat player, it its turn,
        can place one goat on empty spaces on the board until 20 goats are 
        placed.
        
        The tiger player, in its turn, can make a legal move or a legal 
        jump on the board. A legal jump can kill a goat.
        
        The goal for the tiger player is to kill 6 goats and the goal for 
        the goat player is to prevent all tigers from making a move.
        """
        self.board = np.zeros((5, 5), dtype=int)

        # set initial tiger positions
        self.board[0, 0] = 1
        self.board[4, 0] = 1
        self.board[0, 4] = 1
        self.board[4, 4] = 1

        # also keep track of tiger locations for faster checking of winner
        self.tigers = {1: [0, 0], 2: [4, 0], 3: [0, 4], 4: [4, 4]}

        # number of goats remaining to place and number of goats on board
        self.goats_onboard = 0
        self.goats_eaten = 0
        self.goat_eaten = 0
        self.goats_outside = 20
        self.last_goat_row = -1
        self.last_goat_col = -1

        # initial turn is of goat (-1); turn = 1 for tiger; turn = 0 game complete
        self.turn = -1

    def validmove(self, r1, c1, r2, c2):
        """Check if a piece can be moved from p1=[r1,c1] to p2=[r2,c2]

        note: this function does not check if there are pieces on p1 or p2
        """
        distsq = distance_square(r1, c1, r2, c2)

        if (distsq < 1 or distsq >= 4):
            return 0

        if ((r1 + c1) % 2 == 0 or distsq == 1):
            return 1

        return 0

    def validjump(self, r1, c1, r2, c2):
        """Check if a piece can jump from p1=[r1,c1] to p2=[r2,c2]
        
        note: this function checks if there is a goat piece at the midpoint
        but does not check if there is tiger at p1 and if p2 is empty
        """
        midrow = int((r1 + r2) / 2)
        midcol = int((c1 + c2) / 2)

        if (self.board[midrow, midcol] != -1):
            return 0

        if (distance_square(r1, c1, r2, c2) == 4 * distance_square(r1, c1, midrow, midcol)):
            if (self.validmove(r1, c1, midrow, midcol) and self.validmove(midrow, midcol, r2, c2)):
                return 1

        return 0

    @property
    def checkwinner(self):
        """Check if tiger/goat is a winner or the game has no winner yet
        in the process if a tiger is found that cannot be moved, remove it from the board
        """
        if (self.goats_eaten > 4):
            # winner is tiger
            return 1

        # loop over possibilities if the tiger can make a valid move/jump
        # if yes, there is no winner
        # if there are no possibilities then goat wins

        totaltigermoves = 0

        for tiger in self.tigers:
            r1, c1 = self.tigers[tiger]
            tigermoves = 0
            for r2 in range(5):
                for c2 in range(5):
                    if (self.board[r2, c2] == 0):
                        if (self.validmove(r1, c1, r2, c2) == 1 or self.validjump(r1, c1, r2, c2) == 1):
                            tigermoves = tigermoves + 1

            totaltigermoves = totaltigermoves + tigermoves

        if (totaltigermoves > 0):
            return 0

        return -1

    def movetiger(self, tiger, r2, c2):
        """Check if the tiger can be moved from [r1,c1] (location of tiger) to [r2,c2]
        if yes, make the move and update the board and return 1
        if no, return 0
        """
        r1, c1 = self.tigers[tiger]

        if self.turn != 1 or self.board[r1, c1] != 1 or self.board[r2, c2] != 0:
            return 0

        if self.validmove(r1, c1, r2, c2):
            self.board[r2, c2] = 1
            self.board[r1, c1] = 0
            self.tigers[tiger] = [r2, c2]
            return 1

        if self.validjump(r1, c1, r2, c2):
            self.board[r2, c2] = 1
            self.board[r1, c1] = 0
            self.tigers[tiger] = [r2, c2]
            self.last_goat_row = int((r1 + r2) / 2)
            self.last_goat_col = int((c1 + c2) / 2)
            self.board[self.last_goat_row, self.last_goat_col] = 0
            self.goats_eaten = self.goats_eaten + 1
            self.goats_onboard = self.goats_onboard - 1

            return 1

        return 0

    def placegoat(self, r1, c1):
        """Check if [r1,c1] is empty 
        if yes, and goats_outside>0 place a goat and return 1
        else return 0
        """
        if not (self.board[r1, c1] == 0 and self.goats_outside > 0):
            return 0

        self.board[r1, c1] = -1
        self.goats_outside = self.goats_outside - 1
        self.goats_onboard = self.goats_onboard + 1

        return 1

    def movegoat(self, r1, c1, r2, c2):
        """Check if the goat can be moved from [r1,c1] to [r2,c2]
        if yes, make the move and update the board and return 1
        if no, return 0
        """

        if self.turn != -1 or self.board[r2, c2] != 0:
            return 0

        if r1 < 0:
            self.placegoat(r2, c2)
            return 1
        else:
            if self.validmove(r1, c1, r2, c2):
                self.board[r2, c2] = -1
                self.board[r1, c1] = 0
                return 1
        return 0
