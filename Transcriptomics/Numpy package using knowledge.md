**Introduce**

    #Numpy, short for Numerical python, is the most important basic package for numerical computation in Python today.

    One of the core features of Numpy is the N-dimensional array object (ndarray). nadarry is a fast, flexible container for large data in Python.
    Arrays run mathematical operations on blocks using a scalar-like operation syntax.
    

**Input NumPy**

    #Code
    import numpy as np #the np means numpy, we give a short for numpy.
    
**How to Create Ndarray with different dimensions.**

    #create one dimension ndarray
    **Code**
    datal = [1,2,3,4,5]
    print(type(datal)) #First we create a list.
    **Output**
    <class 'list'>
    **Code**
    #We can use np.array(XXX) to make the type from list convert to ndarray. 
    arrl = np.array(datal)
    print(arrl)
    **Output**
    [1 2 3 4 5]
    **Code**
    print(type(arrl))
    **Output**
    <class 'numpy.ndarray'>
    #If we want to check the shape and dimension of this narray, we can use XXXX.shape and XXXX.ndim.
    **Code**
    print(arrl.shape)
    print(arrl.ndim)
    **Output**
    (5,)
    1
    
 **Create a two dimensions ndarray**
 
    #we need create a list, firstly.
    data2 = [[1,2,3,4],[5,6,7,8]]
    type(data2)
    **Output**
    list
    **Code**
    arr2 = np.array(data2)
    print(arr2)
    print(type(arr2))
    print(arr2.shape)
    print(arr2.ndim)
    **Output**
    [[1 2 3 4]
    [5 6 7 8]]
    <class 'numpy.ndarray'>
    (2, 4)
    2
    
 **We also can use function to create specifical array**
 
    #there are some samples
    **Code**
    print(np.zeros(5))
    print(np.ones((2,3)))
    print(np.arange(2,11,3)) #np.arrange function will create a series of front-closed and back-open numbers.
    **Output**
    [0. 0. 0. 0. 0.]
    [[1. 1. 1.]
    [1. 1. 1.]]
    [2 5 8]
    
 **The list of order for ndarray**
 
               
![14](https://user-images.githubusercontent.com/103999272/235300690-af8b6ec7-eea3-430b-908e-46b67270125c.png)
