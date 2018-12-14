#! python3
# -*- coding: utf-8 -*-


import sketch_coding_bf

import numpy as np

def test():
	com_ratio = 0.9
	shape1 = 64
	shape2 =64
	binary_value = 1


	# M = np.random.uniform(-half_base_mask * float_step, (half_base_mask - 1) * float_step, size=(shape1, shape1))

	M = np.load( "to_yang.npy" )
	print(M)

	M1 = sketch_coding_bf.sketch_transform(M, com_ratio, binary_value)
	print(M1)

if __name__ == '__main__':
	test()