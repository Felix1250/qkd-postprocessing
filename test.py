import numpy as np

arr = np.array(range(10))
print(arr)
arr2 = arr.copy()
np.random.shuffle(arr2) 
print(arr)
print(arr2[:3])
print(arr2[3:])
    