import numpy as np
import pandas as pd


nomes = {
	'stomach': 'Estômago',
	'large_bowel': 'Intestino Grosso',
	'small_bowel': 'Intestino Delgado',
}

def decode_rle(mask_rle, shape):
	'''
	mask_rle: run-length as string formated (start length)
	shape: (height,width) of array to return
	Returns numpy array, 1 - mask, 0 - background
	'''
	if not isinstance(mask_rle, str):
			img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
			return img.reshape(shape).T

	s = mask_rle.split()
	starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
	starts -= 1
	ends = starts + lengths
	img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
	for lo, hi in zip(starts, ends):
			img[lo:hi] = 1
	return img.reshape(shape).T

def save_dataframe_as_feather(df: pd.DataFrame, filename: str):
	df.reset_index().drop(columns=['index']).to_feather(filename)