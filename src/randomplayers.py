import numpy as np

def randomplayerTiger(game):
    """ return a random next valid move given the game
    """
    moves = []
    
    for tiger in game.tigers:
        r1,c1 = game.tigers[tiger]
        for r2 in range(5):
            for c2 in range(5):
                if game.board[r2,c2] == 0:
                    if game.validmove(r1,c1,r2,c2) or game.validjump(r1,c1,r2,c2):
                        moves.append([tiger,r2,c2])
    n = len(moves)
    
    if (n>0):
        move = moves[np.random.randint(n)]
        return move
    
    return None

def randomplayerGoat(game):
    """ return a random next valid move given the game
    """
    moves = []
    
    # check placement moves first
    if (game.goats_outside>0):            
        # find the first goat with negative values and move it
        
        for r2 in range(5):
            for c2 in range(5):
                if game.board[r2,c2]==0:
                    moves.append([-1,-1,r2,c2])
        
        n = len(moves)
        
        if (n>0):
            move = moves[np.random.randint(n)]
            return move
        else:
            return None
    
    for r1 in range(5):
        for c1 in range(5):
            if game.board[r1,c1]==-1:
                for r2 in range(5):
                    for c2 in range(5):
                        if (game.board[r2,c2]==0 and game.validmove(r1,c1,r2,c2)==1):
                            moves.append([r1,c1,r2,c2])
    
    n = len(moves)
    
    if (n>0):
        move = moves[np.random.randint(n)]
        return move
    
    return None
