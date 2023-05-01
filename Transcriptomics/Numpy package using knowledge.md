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
     
     **Code**
     #use np.arange() function to creat a ndarray.
     arrld = np.arange(10)
     arrld
     **Output**
     array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
     # if we want to select one of elements in the array, we just use [] to do.
     **Code**
     arrld[3]
     **Output**
     3
          # if we want to get a set of elements in the array.
      **Code**
      arrld_slice = arrld[3:6]
      arrld_slice
      **Output**
      array([3, 4, 5])
      #Numpy's slices are "references", not copies, and modifying a slice modifies the original array.
      # The arrld_slice we just defined is part of the arrld array, and if we make the elements in arrld_slice change, the arrld array will be also changed.
      **Code**
      arrld_slice[0] = 33#that means the arrld_slice array' first element will be defined as 33.
      arrld# same time, the arrld array' fourth element is changed as 33.
      **Output**
      array([ 0,  1,  2, 33,  4,  5,  6,  7,  8,  9])
      # if we don't want to change the original array while changing the sliced array, we can use XXXX[3:6].copy() function to get a copy slice array.
      arrld_slice_copy = arrld[3:6].copy()
     arrld_slice_copy[0] = 333
     print(arrld)
     arrld_slice_copy
     # we can see that only copy array is changed.
     **Output**
     [ 0  1  2 33  4  5  6  7  8  9]
     array([333,   4,   5])
     
**Numpy manipulation of high-dimensional arrays**

     #Firstly, create a senconde dimensional array.
     **Code**
     arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
     arr2d
     **Output**
     array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
     **Code**
     arr2d[0]# if we want to get first row, just use [0]
     **Output**
     array([1, 2, 3])
     **Code**
     arr2d[0][0]# if we want to get first element of row and column, we can use [0][0]
     #output: 1
     **Code**
     arr2d[:,2]# if we want to get thirld column, we can use XXX[:,2]
     #Output: array([3, 6, 9])
     **Code**
     arr2d[0:2,0:2]
     #Output: array([[1, 2],
                    [4, 5]])
                    
**Boolean index (used when performing feature filtering)**
    
     **Code**
     arr2d == 5
     # Output: 
     array([[False, False, False],
       [False,  True, False],
       [False, False, False]])
       
     # We can set conditions to make logical judgments, and then we can filter out the elements that meet 
     the conditions, and we can use.
     **Code**
     arr_features = np.array(["A","B","C","D","E"])# this array represents original data array.
     arr_bool = np.array([True,True,False,False,False]) # This array demonstrates a logical result array.
     arr_features[arr_bool]# It is possible to filter the elements that are Ture in the logical array at the corresponding positions in the original array.
     **Output**
     array(['A', 'B'], dtype='<U1')
     

**Array Transformation**
    
    # Array convert
    **Code**
    arr2d.T #The columns convert to raws.
    **Outpt**
    array([[1, 4, 7],
       [2, 5, 8],
       [3, 6, 9]])
    
    #Number axis conversion
    #Arrays for axis replacement
    #XXX.transpose(()) function can be used to perform axis replacement when processing image dimensions. 
    #If () is of the form (1, 0), it is a row transformation.
    **Code**
    arr2d.transpose((1,0))
    **Output**
    array([[1, 4, 7],
       [2, 5, 8],
       [3, 6, 9]])
    
    # How to make the high-dimension convert to one-dimension.
    # We can use XXXX.flatten() function to get.
    **Code**
    # The two-dimension convert to one-dimension
    arr_flatten = arr2d.flatten()
    arr_flatten
    **Output**
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])
 
    # How to make the low-dimension object converting to high-dimension object
    # We can use the XXX.reshape(,) to get a two-dimension object.
    **Code**
    arr_flat_2d = arr_flatten.reshape(3,3)
    arr_flat_2d
    **Output**
    array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
     
 **Array functions for elements**
        
        #This syntax will do same with every element in the array.
 

![16](https://user-images.githubusercontent.com/103999272/235413798-25592fe0-f2b2-47ad-bafc-62a712e73d85.png)

**General syntax of calculation in the array**

![17](https://user-images.githubusercontent.com/103999272/235414104-42e84933-2500-443a-8560-308490906f39.png)

**About the practice of computing commands**

![18](https://user-images.githubusercontent.com/103999272/235415250-244f5dc9-723e-47f9-822a-2bffe0533e90.png)

![19](https://user-images.githubusercontent.com/103999272/235415934-51f9fe02-6661-4ac8-9d79-59bc55bd3a24.png)
