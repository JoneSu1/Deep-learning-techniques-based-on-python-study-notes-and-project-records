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
    
 **The list of syntax for ndarray**
 
               
![14](https://user-images.githubusercontent.com/103999272/235300690-af8b6ec7-eea3-430b-908e-46b67270125c.png)

**The data type of ndarray**

![15](https://user-images.githubusercontent.com/103999272/235301011-9bea6d78-f69a-4d8e-9d17-3a09a67c62e5.png)

**How to convert the type of ndarray**
    
    #If we want to get a float64 ndarra, we can use XXX.astype(np.float32) to convert the type of ndarray's int32 
    **Code**
    print(arrl.dtype)# output:int32
    float_arr = arrl.astype(np.float32)
    float_arr.dtype #output: dtype('float32')
    
**Array calculation**

     #The rules of array computation are the same as those of simple computation, as long as the  rows and columns of two arrays are the same, 
     you can add, subtract, multiply and divide between two arrays directly.
     **Code**
     a_arr = np.full((2,3),3) #An array of two rows and three columns of all 3s
     a_arr
     **output**
     array([[3, 3, 3],
       [3, 3, 3]])
     **Code**
     b_arr = np.full_like(a_arr,1) #np.full_like(XXXX,) means you can creat a style like XXXX new array.
     b_arr
     **Output**
     array([[1, 1, 1],
       [1, 1, 1]])
     
     **Code, Arithmetic between arrays and arrays **
     a_arr + b_arr
     **Output**
     array([[4, 4, 4],
       [4, 4, 4]])
      
     **Arithmetic between arrays and numbers**
     **Code**
     a_arr * 4
     **Output**
     array([[12, 12, 12],
       [12, 12, 12]])
       **Code**
       6/a_arr
       **Output**
       array([[2., 2., 2.],
       [2., 2., 2.]])
       **Code**
       a_arr > b_arr #If we compare two array, we will get logical results(One by one matching)
       **Output**
       
        array([[ True,  True,  True],
            [ True,  True,  True]])
            
**Indexing and slicing in Numpy**
