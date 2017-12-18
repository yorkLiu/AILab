import numpy as np


# view more numpy examples please
# visit: https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

# np.arange(15) create 15 items from 0 index (0, 1, 2, ...14)
# reshape is convert the array to 3 rows and 5 columns matrix
a = np.arange(15).reshape(3, 5)
print a

print a.shape

print a.ndim

print a.dtype
print a.itemsize
print a.size
print type(a)

b = np.array([4, 5, 6, 7])
c = b.astype(dtype=np.float64)
print c, c.dtype


d = np.zeros((3,4), dtype=np.int32)
print d


# create a array from start index 10, end to 30 and the step is 5
e = np.arange(10, 30, 5)
print e

# create a array from 0 and the max is 2, will generate 10 items
f = np.linspace(0, 2, 10)
print f


a1 = np.random.random((3, 4))
print "a1:",a1
print a1.max(axis=0)
print a1.max(axis=1)


a2 = 2**np.arange(1, 11)
print a2
i2 = a2 == 512
print i2
print a2[i2]

a2.shape=(5,2)
print a2

matrix = np.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])

second_column_25 = matrix[:,1] == 25
print second_column_25
print matrix[second_column_25,:]
print matrix[second_column_25,1]
