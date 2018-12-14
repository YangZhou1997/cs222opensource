#! python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import ctypes
from myhash import myhash1, myhash2, myhash3

def sketch_update(sketch, sketch_len, index_i, index_j):
	hash_value = myhash1(index_i, index_j) % sketch_len
	sketch[hash_value] = 1
	hash_value = myhash2(index_i, index_j) % sketch_len
	sketch[hash_value] = 1
	hash_value = myhash3(index_i, index_j) % sketch_len
	sketch[hash_value] = 1

# M is the original matrix waiting for encoding into the hash array -- the return value. 
# base is the number of bits used in M
# com_ratio is the compression ratio.
def sketch_encoding(M, com_ratio):

	M_int = (M > 0)

	i_len = M_int.shape[0]
	j_len = M_int.shape[1]

	sketch_len = int(i_len * j_len * com_ratio + 1)

	sketch = np.zeros((sketch_len,), dtype=np.int32)

	for i in range(0, i_len):
		for j in range(0, j_len):
			if M_int[i][j] == True:
				sketch_update(sketch, sketch_len, i, j)

	return sketch

def sketch_getvalue(sketch, sketch_len, index_i, index_j):
	hash_value1 = myhash1(index_i, index_j) % sketch_len
	hash_value2 = myhash2(index_i, index_j) % sketch_len
	hash_value3 = myhash3(index_i, index_j) % sketch_len
	return min(sketch[hash_value1], sketch[hash_value2], sketch[hash_value3])


# sketch/2/indicator -> np.int32, M -> np.float32
def sketch_decoding(sketch, i_len, j_len, com_ratio):

	M = np.zeros((i_len, j_len), dtype=np.float32)
	sketch_len = int(i_len * j_len * com_ratio + 1)

	for i in range(0, i_len):
		for j in range(0, j_len):
			if sketch_getvalue(sketch, sketch_len, i, j) == 1:
				M[i][j] = 1
			else: 
				M[i][j] = -1
	return M


def sketch_transform(M, com_ratio):
	sketch = sketch_encoding(M, com_ratio)
	M1 = sketch_decoding(sketch, M.shape[0], M.shape[1], com_ratio)
	return M1


def test():
	com_ratio = 0.9
	shape1 = 64
	shape2 =64


	# M = np.random.uniform(-half_base_mask * float_step, (half_base_mask - 1) * float_step, size=(shape1, shape1))

	M = np.load( "to_yang.npy" )
	print(M)

	M1 = sketch_transform(M, com_ratio)
	print(M1)

if __name__ == '__main__':
	test()