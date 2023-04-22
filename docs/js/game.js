dim = 100;
tigerRadius = 20;
goatRadius = 15;
whichPieceX = -1;
whichPieceY = -1;
leftPad = 15;
topPad = 15;

function game(n) {

	pieces = [[1, 0, 0, 0, 1],
	          [0, 0, 0, 0, 0],
			  [0, 0, 0, 0, 0],
			  [0, 0, 0, 0, 0],
			  [1, 0, 0, 0, 1]];
	tigers = { 1: [0,0], 2: [4,0], 3: [0,4], 4: [4,4] };
    whichGame = n;
    goatsEaten = 0;
    goatsOnhand = 20;
    goatOnhand = 0;
    goatOnhandRow = 0;
    goatOnhandCol = 0;
    tigerOnhand = 0;
    tigerOnhandRow = 0;
    tigerOnhandCol = 0;
    turn = 2;
    
    canvas = document.getElementById("canvas");
    ctx = canvas.getContext("2d");
    drawBoard();
    drawPieces();
}
	
function drawLine(x1, y1, x2, y2) {
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}

function drawBoard() {
    ctx.beginPath();
    boardLeft = leftPad + tigerRadius;
    boardTop = topPad + tigerRadius;
    ctx.fillStyle="white";
    ctx.fillRect(boardLeft-tigerRadius, boardTop-tigerRadius, dim*4+tigerRadius*2, dim*4+tigerRadius*2);
    ctx.strokeStyle="darkblue";
    ctx.strokeRect(boardLeft, boardTop, dim*4, dim*4);
    ctx.moveTo(boardLeft, boardTop);
    for (i = 0; i < 5; ++i) {
	drawLine(boardLeft, boardTop + i * dim, boardLeft + dim *4, boardTop + i * dim);
	drawLine(boardLeft + i * dim, boardTop, boardLeft + i * dim, boardTop + dim *4);
    }
    drawLine(boardLeft, boardTop, boardLeft+dim*4, boardTop+dim*4);
    drawLine(boardLeft+dim*4, boardTop, boardLeft, boardTop+dim*4);
    ctx.moveTo(boardLeft, boardTop+dim*2);
    ctx.lineTo(boardLeft+dim*2, boardTop);
    ctx.lineTo(boardLeft+dim*4, boardTop+dim*2);
    ctx.lineTo(boardLeft+dim*2, boardTop+dim*4);
    ctx.lineTo(boardLeft, boardTop+dim*2);
    ctx.stroke();
}
			
function drawPieces() {
    ctx.strokeStyle="white";
    for (i = 0; i < 5; ++i) {
	for (j = 0; j < 5; ++j) {
	    if (pieces[i][j]==1) {
		ctx.beginPath();
		ctx.arc(boardLeft+dim*j, boardTop+dim*i, tigerRadius,0, 2*Math.PI, true);
		ctx.moveTo(boardLeft+dim*j-2, boardTop+dim*i);
		ctx.fillStyle = "rgba(200,0,0,0.96)";
		ctx.fill();
	    }
	    else if (pieces[i][j]==2) {
		ctx.beginPath();
		ctx.arc(boardLeft+dim*j, boardTop+dim*i, goatRadius, 0, 2*Math.PI, true);
		ctx.moveTo(boardLeft+dim*j-2, boardTop+dim*i);
		ctx.fillStyle = "rgba(10,200,100,0.96)";
		ctx.fill()
	    }
	}
    }
    
    eaten.value = goatsEaten;
    onhand.value = goatsOnhand;
}

function mouseup() {
    drawBoard();
    drawPieces();
    if (tigerOnhand!=0 || goatOnhand!=0) {
	clickBoard();
    }
}

function clickBoard() {
    // check which piece is clicked;
    for (i = 0; i < 5; ++i) {
	for (j = 0; j < 5; ++j) {
	    posX = leftPad+ tigerRadius + j * dim;
	    posY = topPad+tigerRadius + i * dim;
	    if ((mouseX>posX-tigerRadius) && (mouseX<posX+tigerRadius) && (mouseY>posY-tigerRadius) && (mouseY<posY+tigerRadius)) {
			whichPieceX = i;
			whichPieceY = j;
	    	}
		}
    }
    
    // check for the game turn, make a move, && change the turn 
    if (whichPieceX>-1) {
	if (whichGame==1 || whichGame==2) {
	    if (turn==2 && goatsOnhand>0 && pieces[whichPieceX][whichPieceY]==0) {
			goatsOnhand = goatsOnhand-1;
			pieces[whichPieceX][whichPieceY]=2;
			turn = 1;
	    	}
	    else if (turn==2 && goatsOnhand==0 && pieces[whichPieceX][whichPieceY]==2) {
			goatOnhand = 1;
			goatOnhandRow = whichPieceX;
			goatOnhandCol = whichPieceY;
	    }
	    else if (turn==2 && goatOnhand==1 && pieces[whichPieceX][whichPieceY]==0) {
			if (validmove(goatOnhandRow, goatOnhandCol, whichPieceX, whichPieceY)==1) {
		    pieces[whichPieceX][whichPieceY]=2;
		    pieces[goatOnhandRow][goatOnhandCol]=0;
		    goatOnhand = 0;
		    turn = 1;
			}
	    }
	}
	if (whichGame==1 || whichGame==3) {
	    if (turn==1 && pieces[whichPieceX][whichPieceY]==1){
			tigerOnhand = 1;
			tigerOnhandRow = whichPieceX;
			tigerOnhandCol = whichPieceY;
	    }
	    if (turn==1 && tigerOnhand==1 && pieces[whichPieceX][whichPieceY]==0) {
		if (validmove(tigerOnhandRow, tigerOnhandCol, whichPieceX, whichPieceY)==1) {
		    pieces[whichPieceX][whichPieceY]=1;
		    pieces[tigerOnhandRow][tigerOnhandCol]=0;
		    tigerOnhand = 0;
		    turn = 2;
		}
		else if (validjump(tigerOnhandRow, tigerOnhandCol, whichPieceX, whichPieceY)==1) {
		    pieces[whichPieceX][whichPieceY]=1;
		    pieces[tigerOnhandRow][tigerOnhandCol]=0;
		    pieces[(whichPieceX+tigerOnhandRow)/2][(whichPieceY+tigerOnhandCol)/2]=0;
		    tigerOnhand = 0;
		    turn = 2;
		    goatsEaten = goatsEaten + 1;
			}
	    }
	}
    }

    if (whichGame==2 && turn==1) {
	//bestMoveTiger();
		tigernetPlayer();
    }
    
    // check piece code end
    // at last set the clicked piece to none
    whichPieceX = -1;
    whichPieceY = -1;
    drawBoard();
    drawPieces();

    if (whichGame!=0) {
		winner = checkWinner();
		if (winner==1) {
	    	alert("Tiger Wins");
	    	whichGame = 0;
		}
		else if (winner==2) {
	    	alert("Goat Wins");
	    	whichGame = 0;
		}
    }
}

function checkWinner() {
    if (goatsEaten>5) {
		return 1;
    }
    
    for(r1=0;r1<5;++r1){
		for (c1=0; c1<5; ++c1) {
	    	if (pieces[r1][c1]==1) {
				for (r2=0; r2<5; ++r2) {
		    		for (c2=0; c2<5; ++c2) {
						if (pieces[r2][c2]==0) {
			    			if (validmove(r1, c1, r2, c2)==1 || validjump(r1, c1, r2, c2)==1) {
								return 0;
			    			}
						}
		    		}
				}
	    	}
		}
    }
    return 2;
}