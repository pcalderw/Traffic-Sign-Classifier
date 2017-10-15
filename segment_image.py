import numpy as np
###
# This is used as a module because running it in a cell causes the jupyters kernel to crash
###
def segment(X, y):
	X_train_tl = X[:, :24, :24]
	X_train_tr = X[:, 8:, :24]
	X_train_bl = X[:, :24, 8:]
	X_train_br = X[:, 8:, 8:]
	X_train_c  = X[:, 2:26, 2:26]
	X_t = np.concatenate((X_train_tl, X_train_tr, X_train_bl, X_train_br, X_train_c))
	y_t = np.concatenate((y,y,y,y,y))
	return X_t, y_t

###
# This will retun X but each example the center segment
###
def center_segment(X):
	return X[:, 2:26, 2:26]

