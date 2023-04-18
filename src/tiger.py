import torch
import torch.nn as nn
import numpy as np

class TigerNet(nn.Module):
    """ define the neural network for TigerNet
    """
    def __init__(self):
        super(TigerNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 2)
        self.conv2 = nn.Conv2d(50, 100, 3)
        self.convTrans = nn.ConvTranspose2d(100, 4, 4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = (self.conv2(x))
        x = self.sigmoid(self.convTrans(x))
        return (x).view(-1)
        

def tigernetPlayer(game, neuralnet, choice="random"):
    """ return the next goat move using output from the GoatNet nn
        zero out the probabilities for illegal moves in the process of 
        choosing the next move
    """
    
    b = game.board
    b3 = np.expand_dims(np.stack((b==0,b==1,b==-1)), 0).astype('f4')
    btensor = torch.from_numpy(b3).to("cpu")
    output = neuralnet(btensor)

    probs = output.detach().cpu().numpy()

    # filter out illegal moves by setting the probabilities to zero
    probs = np.clip(probs, 0.0001, 0.9999)
    ind = 0
    for tiger in game.tigers:
        r1, c1 = game.tigers[tiger]
        for r2 in range(5):
            for c2 in range(5):
                if not(game.validmove(r1,c1,r2,c2)==1 or game.validjump(r1,c1,r2,c2)==1):
                    probs[ind] = 0.0
                if (game.board[r2,c2]!=0):
                    probs[ind] = 0.0
                
                ind = ind + 1
    
    if (probs.sum()==0):
        print (game.board)
        print(probs)
        print(game.checkwinner)
        for tiger in game.tigers:
            r1, c1 = game.tigers[tiger]
            for r2 in range(5):
                for c2 in range(5):
                    if ((game.validmove(r1,c1,r2,c2)==1 or game.validjump(r1,c1,r2,c2)==1) and (game.board[r2,c2]==0)):
                        print (r1,c1,r2,c2)
    else:
        probs /= probs.sum()

    if (choice=="best"):
        move = np.argmax(probs)
    else:
        move = np.random.choice(probs.size, p=probs)
    
    # convert back to (row, col)
    tiger = 1+move//25
    r2 = (move-(tiger-1)*25)//5
    c2 = (move%25)%5
        
    return [tiger,r2,c2], output[move]
        
        
