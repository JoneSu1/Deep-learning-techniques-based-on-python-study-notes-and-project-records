**Jupyter notebook base code**
![7](https://user-images.githubusercontent.com/103999272/233595611-414997e6-161a-479b-aa7d-b0b3d863d282.png)
![8](https://user-images.githubusercontent.com/103999272/233595629-4cc638fa-f469-44cb-83c0-9c168df83f0e.png)
![9](https://user-images.githubusercontent.com/103999272/233595658-15955508-e509-45ac-b10a-1194bfb709a4.png)
![10](https://user-images.githubusercontent.com/103999272/233596638-b1f460bf-0d4d-4f53-a3ca-9768ad63e196.png)

    **Base comupting code**
![11](https://user-images.githubusercontent.com/103999272/233596822-2526422e-c41d-4a72-a142-5dd996a1495d.png)

    **Code**
    a = 7
    b = 2
    print(a%b)
    print(a//b)
    output
    1
    3
  
  
    **Scientifical computing code**
![12](https://user-images.githubusercontent.com/103999272/233596988-74a504af-a916-487b-bbef-e73268794914.png)
![13](https://user-images.githubusercontent.com/103999272/233597041-3c6dd400-740e-48d3-aad4-d26eebcfce02.png)

    **code**
    a = -1
    print(abs(a))
    print(max(1,5,10))
    print(round(3.6))
    output
    1
    10
    4
    code this computing is based on the package "math"
    import math
    print(math.fabs(-1.03))
    print(math.floor(1.9))
    print(math.ceil(1.9))
    print(math.exp(0))
    print(math.sqrt(4))
    print(math.log10(10))
    print(math.log(100,10))
    print(math.pow(2,3))
    output
    1.03
    1
    2
    1.0
    2.0
    1.0
    2.0
    8.0


    
    
    
**Regarding str computing**

    **code: The str advoter can be used for Multiplication and addition**
    a = "hello"
    b = "world"
    print(a + b)
    print(a * 3)
    print(a[0])
    a[0]
    a[1:3]
    **output**
    helloworld
    hellohellohello
    h
    'el'
 
**Regarding Control flow**

**1.The statements of Judement**

    Notice:
    An if can be used by itself
    An if, which can be used in conjunction with one or more elifs
    An if, which can be used in conjunction with one or more elifs and an else（end of code flow have to remain an else）
    An if can be used in conjunction with an else

**Code**

    a = 5
    b = 3
    if a < b:
        print("a<b")
    elif a==b:
        print("a==b")
    else:
        print("a>b")
    output:
    a>b

**range**

The range() function will help us get a arithmetic progression.

**Code**

    range(10)
    #if I just use the range() funcation, I will get output: range(0,10)
    list(range(10))
    #if I use list() funcation, I will show: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    list(range(0,10,2))
    #The first number indicates the value at the beginning of the series.
    #The second number represents the maximum value of this series.
    #The third number represents the difference of this series of equal differences.
    #[0, 2, 4, 6, 8]
    
**3. "for" loop**

**Code**

    for i in range(5):
       print(i)
       # means the number of i is in the range (0,1,2,3,4).
       
**Regarding "continue" and "break" in the loop.**

    #continue
    for i in range(5):
       if i % 2 ==0:
          continue
       print(i)
       #If the remainder of i divided by 2 is equal to 0, the cycle continues.
       #When i is at the remainder of 2 is not equal to 0, then go to the next step and output i.

     #break
     for i in range(4):
      for j in range(5,9):
         if j == 6:
               break
         print(i,j)
    # The role of this break is to stop the loop when j equals 6 and output the (i,j) before j equals 6
    # The list() function will collect the vaule of producted by the range() function.
    **output**
    0 5
    1 5
    2 5
    3 5

**while loop**

    # This loop will continue until the qualifying condition is not met.
    **Code**
    x = 10
    while x>0:
      print(x)
      x = x-2
      
    outPut
    8
    6
    4
    2
    0
    # means that when the x <0, this function will stop.
    # We also can use the break to add the condition of stop.
    **Code**
    x = 10
    while x>0:
      print(x)
      x = x-2
      if x<5:
        break
     output
     10
     8
     6
     
 **Regarding List function**
 
    #The length and content of list can be changed.
    #We can use the featurelist[] function to creat a list.
    **Code**
    featurelist = ["mean","median","Energy", "Entropy"]

**Add and remove element in list**

    # We can use the function append to add elements to the end.
    **code**
    featurelist.append("maximum")
    print(featurelist)
    **output**
    ['mean', 'median', 'Energy', 'Entropy', 'maximum', 'maximum']
    # We can also use the insert function to place the element at the specified position.
    **code**
    featurelist.insert(0,"minimum")
    print(featurelist)
    **output**
    ['minimum', 'mean', 'median', 'Energy', 'Entropy', 'maximum', 'maximum']
    # we also can use pop function to remove element at the specified postion. And The return value is the element that was removed.
    **Code**
    featurelist_pop = featurelist.pop(2)
    print(featurelist_pop)
    print(featurelist)
    **output**
    median
    ['minimum', 'mean', 'Energy', 'Entropy', 'maximum', 'maximum']
    # We can use the remove function to remove the first met element.  
    **Code**
    featurelist.remove("mean")
    print(featurelist)
    **output**
    ['minimum', 'Energy', 'Entropy', 'maximum', 'maximum']
    # If we want check a element weather or not involed in this list. we can use the in.
    "Energy" in featurelist
    output: Ture

**How to link two list**

    # We can use addtion to link
    **Code**
    list_A = ["a","b","c"]
    list_B = ["d","e"]
    list_x = list_A + list_B
    print(list_x)
    **output**
    ['a', 'b', 'c', 'd', 'e']
    # we also can use fuction extend to link two list.
    **Code**
    list_A.extend(list_B)
    print(list_A)
    **OUTPUT**
    ['a', 'b', 'c', 'd', 'e']

**How to select specifical element**

   
    **Code**
    print(list_A[0]) # choose first element
    print(list_A[1:3])# choose element 2 and 3
    print(list_A[-1]) #The negative sign indicates that the element is selected from the opposite direction of the list. 
    print(list_A[2:]) # from 3th element to 
    print(list_A[:3])# form beging to the second element.
    print(list_A[-3:-1])# The negative sign indicates an element from the 3rd to the 2nd.
    print(list_A[::2])#  A double colon means that the output is separated by several elements.
    print(list_A[::-1]) 
    **OUTPUT**
    a
    ['b', 'c']
    e
    ['c', 'd', 'e', 'd', 'e']
    ['a', 'b', 'c']
    ['e', 'd']
    ['a', 'c', 'e',]
    ['e', 'd', 'e', 'd', 'c', 'b', 'a']
    

**How is the sorting done at the list?**

    #we can use sort function to define how to orde these elements.
    **code**
    list_C = ["saw","small","He","foxes","six"]
    list_C.sort()# if we don't define the rule of order, It will follow the A-Z.
    print(list_C)
    **output**
    ['He', 'foxes', 'saw', 'six', 'small']
    
    #if we define, we can do that: list_C.sort(key= len)# it will follow the length of elements to order.
    
  **Dictionary**
  
     #A dictionary is a collection of keys and values that can be modified in length and content.
     **Code**
     dict_a = {"Name":"zhao","Age": 16,"Gender":"Male"}# create a dictionary
     dict_a["Name"]# look up the dictionary
     **output**
     'zhao'
 
 **How to insert a element in the dict**
 
    **Code**
    dict_a["Language"] = "English"
    print(dict_a)
    **output**
    {'Name': 'zhao', 'Age': 16, 'Gender': 'Male', 'Language': 'English'}
    
 **How to motify element at dict**
 
    #Motify
    **Code**
    dict_a["Age"] = 18
    print(dict_a)
    **output**
    {'Name': 'zhao', 'Age': 18, 'Gender': 'Male', 'Language': 'English'}
    
    #How to delect element
    #Generaly we can use del function to delect element in dict.
    **code**
    del dict_a["Age"]
    print(dict_a)
    **output**
    {'Name': 'zhao', 'Gender': 'Male', 'Language': 'English'}
    # on the other way, we can use XXX.pop("element") to delect.
    **Code**
    dict_a.pop("Gender")
    print(dict_a)
    **output**
   {'Name': 'zhao', 'Language': 'English'}
   

**If we want to see the value or key in the dictionary separately**

    #we can use the XXXX.keys(), and XXXX.values() to check.
    **Code**
    dict_a.keys()
    **output**
    dict_a.keys()
    #Regarding values
    **Code**
    dict_a.values()
    **Output**
    dict_values(['zhao', 'English'])
    
**How to merge two dictionary?**

    #we can use A_dictionary.update(B_dictionary) to merge A and B dictionary.
    **Code**
    dict_a.update({"ID":123})
    print(dict_a)
    **Output**
    dict_a.update({"ID":123})
    {'Name': 'zhao', 'Language': 'English', 'ID': 123}
