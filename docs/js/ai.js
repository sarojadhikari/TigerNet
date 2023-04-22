/**
 * @author saroj
 */

function validmove(r1, c1, r2, c2) {
    // no check for pieces here...do it manually
    // this is useful because valimove is used for tiger, goat, && ai moves
    
    if (distance(r1, c1, r2, c2)<2) {
	if ((r1+c1)%2==0) {
	    return 1;
	} else {
	    if (distance(r1, c1, r2, c2)==1) {
		return 1;
	    }
	}
    }
    return 0;
}

function validjump(r1, c1, r2, c2) {
    // (midRow, midCol) should be a goat; two end points can have any piece 
    // || none so check it manually while using validjump
    midRow = (r1+r2)/2;
    midCol = (c1+c2)/2;
    if (distance(r1, c1, midRow, midCol)==0.5*distance(r1, c1, r2, c2)) {
	if (isGoat(midRow, midCol) && validmove(r1, c1, midRow, midCol) && validmove(midRow, midCol, r2, c2)) {
	    return 1;
	}
    }
}

async function tigernetPlayer() {
	// the onnx python code works as follows
	// b = bg.board (5x5 numpy array)
	// b3 = np.expand_dims(np.stack((b==0,b==1,b==2)),0).astype('f4')
	// output = session.run(None, {"input.1" : b3})
	// output is an array with 100 probability values that
	// need to be translated to tiger moves
	//const session = await ort.InferenceSession.create('./tigernn.onnx');
	// define arrays for one hot encoding
	//session = await ort.InferenceSession.create('./tigernn.onnx');
	session = await ort.InferenceSession.create('./bg_trained_400_4.onnx');

	var b0 = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]];
	var b1 = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]];
	var b2 = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]];


	for (r=0; r<5; ++r) {
		for (c=0; c<5; ++c) {
			if (pieces[r][c]==0) {
				b0[r][c]=1;
			} else if (pieces[r][c]==1) {
				b1[r][c]=1;
			} else if (pieces[r][c]==2) {
				b2[r][c]=1;
			}
		}
	}

	const inputTensor = new ort.Tensor('float32', [b0.flat(),b1.flat(),b2.flat()].flat(), [1, 3, 5, 5])
	//console.log(inputTensor)
	const output = await session.run( { "input.1": inputTensor});
	console.log(output)
	const probs = output[13].data;
	//console.log(probs)
	// convert the output probabilities to tiger move -- need to track tigers
	ind = 0;
	maxind = 0;

	for (tiger=1; tiger<5; ++tiger) {
		[r1, c1] = tigers[tiger];
		for (r2=0; r2<5; ++r2) {
			for (c2 = 0; c2 < 5; ++c2) {
				if (pieces[r2][c2] != 0) {
					probs[ind] = 0.0;
				}

				if (validmove(r1, c1, r2, c2) != 1 && validjump(r1, c1, r2, c2) != 1) {
					probs[ind] = 0.0;
					//console.log(ind);
				}
				if (probs[ind] > probs[maxind]) {
					maxind = ind;
				}
				//console.log(r1, c1, r2, c2, validmove(r1,c1,r2,c2), validjump(r1,c1,r2,c2));
				ind = ind + 1;
			}
		}
	}

	// get the index based on probability as weights (for randomness)

	const cumWeights = [];
	for (i=0; i<ind; ++i) {
		cumWeights[i] = probs[i] + (cumWeights[i-1] || 0);
	}

	const randomIndex = Math.random() * cumWeights[ind-1];

	for (i=0; i < ind; ++i) {
		if (cumWeights[i] >= randomIndex) {
			index = i;
			break;
		}
	}

	tiger = 1+Math.floor(index/25);
	r2 = Math.floor((index-(tiger-1)*25)/5);
	c2 = (index%25)%5;
	[r1,c1] = tigers[tiger];

	// make the move/jump and update the board
	if (validmove(r1,c1,r2,c2)==1) {
		pieces[r1][c1] = 0;
		pieces[r2][c2] = 1;
		tigers[tiger] = [r2, c2];
	} else if (validjump(r1,c1,r2,c2)==1) {
		pieces[r1][c1] = 0;
		pieces[r2][c2] = 1;
		pieces[(r1+r2)/2][(c1+c2)/2]=0;
		goatsEaten = goatsEaten + 1;
		tigers[tiger] = [r2,c2];
	}
	drawBoard();
	drawPieces();
	turn = 2;
}

function isGoat(row, col) {
    if (Math.round(row)==row && Math.round(col)==col) {
	if ((row>=0 && row <5) && (col>=0 && col<5) && pieces[row][col]==2) {
	    return true;
	}
    }
}

function isBlank(row, col) {
    if (Math.round(row)==row && Math.round(col)==col) {
	if ((row>=0 && row <5) && (col>=0 && col<5) && pieces[row][col]==0) {
	    return true;
	}
    }
}

function distance(x1, y1, x2, y2) {
    return (Math.sqrt(Math.pow((x2-x1),2)+Math.pow((y2-y1), 2)));
}	
