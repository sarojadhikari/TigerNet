from PySide6.QtCore import Qt, QRect, Slot, Signal

from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPalette, QColor, QPainter, QMouseEvent

from .baghchal import Baghchal
from .tiger import TigerNet, tigernetPlayer
from .randomplayers import randomplayerGoat

from torch import set_num_threads
import torch.optim as optim

set_num_threads(5)
net = TigerNet().to("cpu")

def small_lr():
    global optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

def large_lr():
    global optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)

large_lr()

class Board(QWidget):

    value_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._x = 50
        self._y = 40
        self._w = 40
        self._s = 60
        self._eps = 50
        self._nrandom = 0

        self.game = Baghchal()
        self.saved_move_probs = []
        self.saved_goats_eaten_or_not = []
        self.current_game_move_probs = []
        self.current_game_goats_eaten_or_not = []

        self.setPalette(QPalette(QColor(200, 250, 250)))
        self.setAutoFillBackground(True)

        self.goat_hold_r = -1
        self.goat_hold_c = -1

    def newgame(self):
        if (len(self.current_game_move_probs)>0):
            self.saved_move_probs.append(self.current_game_move_probs)
            self.saved_goats_eaten_or_not.append(self.current_game_goats_eaten_or_not)
            self.current_game_move_probs = []
            self.current_game_goats_eaten_or_not = []

        self.game = Baghchal()
        self.update()

    def savemodel(self):
        from numpy import expand_dims, stack
        from torch.onnx import export
        from torch import from_numpy

        nplayed = len(self.saved_move_probs)
        filename = "savedmodels/bg_trained_"+str(self._nrandom)+"_"+str(nplayed)+".onnx"
        b = self.game.board
        b3 = expand_dims(stack((b==0,b==1,b==2)), 0).astype('f4')
        btensor = from_numpy(b3)
        export(net, btensor, filename)

    def train(self):
        self._nrandom = self._nrandom + self._eps
        for eps in range(self._eps):

            net.train()
            winners = []
            episodes = []
            geatens = []
            games = 50

            for game_index in range(games):
                winner, move_probs, geaten = self.play_against_random()
                winners.append(winner)
                episodes.append(move_probs)
                geatens.append(geaten)

            net.zero_grad()
            total_loss = 0.0

            for geatens, move_probs in zip(geatens, episodes):
                total_goats_eaten = sum(geatens)
                for move_prob, geaten in zip(move_probs, geatens):
                    prob = move_prob

                    if (geaten != 1):
                        prob = 1 - prob

                    if (prob < 0.01):
                        prob = prob + 0.01

                    loss = -prob.log() - total_goats_eaten / games
                    loss.backward()

                    total_loss += loss.detach().to("cpu")

            optimizer.step()
            self.newgame()
            print (winners.count(0), winners.count(1), total_loss)
        return winners, total_loss

    def train_with_saved_games(self):
        net.train()
        net.zero_grad()
        total_loss = 0.0
        print (self.saved_move_probs)
        for geatens, move_probs in zip(self.saved_goats_eaten_or_not, self.saved_move_probs):
            total_goats_eaten = sum(geatens)
            for move_prob, geaten in zip(move_probs, geatens):
                prob = move_prob

                if (geaten != 1):
                    prob = 1 - prob

                if (prob < 0.01):
                    prob = prob + 0.01

                loss = -prob.log() - total_goats_eaten / len(self.saved_move_probs)
                loss.backward(retain_graph=True)

                total_loss += loss.detach().to("cpu")

        optimizer.step()
        print (total_loss)
        self.saved_goats_eaten_or_not = []
        self.save_move_probs = []

        return 1

    def play_against_random(self):
        """play a game against random goat moves
        """
        self.newgame()
        move_probs = []
        goat_eaten_or_not = []
        MAX_STEPS = 25
        steps = 0
        winner = 0
        last_goats_eaten = 0

        while (winner == 0 and steps < MAX_STEPS):
            steps = steps + 1
            if (self.game.turn == -1):
                move = randomplayerGoat(self.game)
                if not(move):
                    print (self.game.board)
                    winner = 1
                    break
                self.game.movegoat(move[0], move[1], move[2], move[3])
                winner = self.game.checkwinner
            else:
                move, move_prob = tigernetPlayer(self.game, net)
                self.game.movetiger(move[0], move[1], move[2])
                move_probs.append(move_prob)

                if (self.game.goats_eaten>last_goats_eaten):
                    goat_eaten_or_not.append(1)
                    last_goats_eaten = self.game.goats_eaten
                else:
                    goat_eaten_or_not.append(0)

            winner = self.game.checkwinner
            current_turn = self.game.turn
            self.game.turn = -current_turn

        return winner, move_probs, goat_eaten_or_not

    @Slot(int)
    def set_expisodes(self, w):
        self._eps = w
        self.value_changed.emit(self._eps)

    def mousePressEvent(self, event: QMouseEvent):
        lastclick = event.position()
        r, c = self.pos2piece(lastclick.x(), lastclick.y())

        if (r > -1 and c > -1):
            if (self.game.turn == -1 and self.game.board[r,c] == -1):
                self.goat_hold_r = r
                self.goat_hold_c = c

            if (self.game.turn == -1 and self.game.board[r,c] == 0):
                if (self.game.goats_outside>0):
                    #print ("here")
                    self.game.placegoat(r,c)
                    self.game.turn = 1
                elif (self.goat_hold_r > -1):
                    r1 = self.goat_hold_r
                    c1 = self.goat_hold_c

                    if (self.game.validmove(r1, c1, r, c)):
                        self.game.movegoat(r1, c1, r, c)
                        self.game.turn = 1

                    self.goat_hold_r = -1
                    self.goat_hold_c = -1


        winner = self.game.checkwinner

        if (self.game.turn == 1 and winner != -1):
            move, move_prob = tigernetPlayer(self.game, net, choice="random")
            success = self.game.movetiger(move[0], move[1], move[2])

            if (success):
                self.current_game_move_probs.append(move_prob)
                print (move_prob)
                self.game.turn = -1

                if (self.game.goat_eaten):
                    self.current_game_goats_eaten_or_not.append(1)
                    self.game.goat_eaten = 0
                else:
                    self.current_game_goats_eaten_or_not.append(0)

        self.update()

    def pos2piece(self, x, y):
        """
        convert the position (x,y) to piece (row, col)

        both row and col must be positive values for a click
        to be a valid piece on board
        """
        row = -1
        col = -1

        for i in range(5):
            if (y > self._y+i*self._s and y < self._y+i*self._s+self._w):
                row = i
                break
        for j in range(5):
            if (x > self._x+j*self._s and x < self._x+j*self._s+self._w):
                col = j
                break

        return row, col

    def paintEvent(self, event):

        TigerColor = QColor(255, 0, 0, 120)
        GoatColor = QColor(0, 0, 255, 80)

        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(Qt.black)
            for i in range(5):
                for j in range(5):
                    if (self.game.board[i,j]==1):
                        painter.setBrush(TigerColor)
                    elif (self.game.board[i,j]==-1):
                        painter.setBrush(GoatColor)
                    else:
                        painter.setBrush(Qt.white)

                    painter.drawEllipse(QRect(self._x+j*self._s, self._y+i*self._s,
                    self._w, self._w))

            # draw board lines
            painter.setPen(Qt.darkCyan)
            painter.setBrush(Qt.darkCyan)

            dw = self._w/2
            painter.drawLine(self._x+dw, self._y + dw,
                             self._x+4*self._s+dw, self._y+4*self._s+dw)
            painter.drawLine(self._x+dw, self._y + self._s*4 + dw,
                             self._x+4*self._s+dw, self._y+dw)

            for i in range(5):
                painter.drawLine(self._x+i*self._s+dw, self._y+dw,
                                 self._x+i*self._s+dw, self._y+4*self._s+dw)
                painter.drawLine(self._x+dw, self._y+i*self._s+dw,
                                 self._x+4*self._s+dw, self._y+i*self._s+dw)

            painter.drawLine(self._x+dw, self._y+2*self._s+dw,
                             self._x+2*self._s+dw, self._y+dw)
            painter.drawLine(self._x+4*self._s+dw, self._y+2*self._s+dw,
                             self._x+2*self._s+dw, self._y+dw)
            painter.drawLine(self._x+4*self._s+dw, self._y+2*self._s+dw,
                             self._x+2*self._s+dw, self._y+4*self._s+dw)
            painter.drawLine(self._x+2*self._s+dw, self._y+4*self._s+dw,
                             self._x+dw, self._y+2*self._s+dw)


            # text information
            info1 = "Goats Outside: " + str(self.game.goats_outside)
            info2 = "Goats Eaten: " + str(self.game.goats_eaten)
            info = info1 + " " + info2
            painter.drawText(50, 5, 250, 50, 0, info)











