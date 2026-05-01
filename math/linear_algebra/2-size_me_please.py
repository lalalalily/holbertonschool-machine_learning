#!/usr/bin/env python3
def matrix_shape(matrix):
    a = 0
    for i in matrix:
        b = 0
        for j in i:
            b+= 1
        a+= 1
    if type(j) == list:
        c = len(j)
        return[a,b,c]
    return[a,b]