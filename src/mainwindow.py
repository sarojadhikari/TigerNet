# This Python file uses the fol
from PySide6.QtCore import Qt

from PySide6.QtWidgets import QApplication, QPushButton, QWidget
from PySide6.QtWidgets import QGridLayout, QSlider, QLCDNumber
from PySide6.QtGui import QFont

from .board import Board, large_lr, small_lr

class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        quit_button = QPushButton("Quit")
        quit_button.setFont(QFont("Times", 18, QFont.Bold))

        train_button = QPushButton("Train")
        train_button.setFont(QFont("Times", 18, QFont.Bold))
        train_with_saved_button = QPushButton("Train with played games")
        train_with_saved_button.setFont(QFont("Times", 18, QFont.Bold))
        newgame_button = QPushButton("New game")
        newgame_button.setFont(QFont("Times", 18, QFont.Bold))
        lrdecrease_button = QPushButton("LR = 0.0001")
        lrincrease_button = QPushButton("LR = 0.001")
        savemodel_button = QPushButton("Save model")

        slider = QSlider(Qt.Horizontal)
        slider.setRange(10, 101)
        slider.setValue(50)

        gameBoard = Board()
        lcd = QLCDNumber(2)
        lcd.display(50)

        # connect the click events to respective functions
        train_button.clicked.connect(gameBoard.train)
        train_with_saved_button.clicked.connect(gameBoard.train_with_saved_games)
        newgame_button.clicked.connect(gameBoard.newgame)
        quit_button.clicked.connect(QApplication.quit)
        savemodel_button.clicked.connect(gameBoard.savemodel)
        lrdecrease_button.clicked.connect(small_lr)
        lrincrease_button.clicked.connect(large_lr)
        slider.valueChanged.connect(lcd.display)
        slider.valueChanged.connect(gameBoard.set_expisodes)


        # add all the widgets on the grid layout
        layout = QGridLayout(self)
        layout.addWidget(quit_button, 0, 0)
        layout.addWidget(newgame_button, 0, 1)
        layout.addWidget(lcd, 1, 0)
        layout.addWidget(slider, 2, 0)
        layout.addWidget(gameBoard, 1, 1)
        layout.addWidget(train_button, 3, 0)
        layout.addWidget(train_with_saved_button, 3, 1)
        layout.addWidget(savemodel_button, 4, 1)
        layout.addWidget(lrdecrease_button, 4, 0)
        layout.addWidget(lrincrease_button, 5,0)

        layout.setColumnStretch(1, 5)
