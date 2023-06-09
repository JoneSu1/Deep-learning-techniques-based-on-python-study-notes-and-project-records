我将使用谷歌提供的免费算力环境Golab配置进行深度学习的环境.

Golab是已经配置好了的Gpu训练环境。 
我们可以现在本地写好代码然后再上传到Golab进行操作.


**在外部云端服务器使用Gpu计算服务**
**Linux常用命令**

   ls(list) ls
   命令用于列举当前工作目录的文件或文件夹
   cd(change directory) cd
   命令用于切换文件路径
   pwd(print working directory) pwd
   命令用于显示当前
   工作目录
   mkdir(make directory)
   
   rmdir( remove diretory)
   rm（remove)
   cp(copy) (cp 需要拷贝的文件名字 拷贝到的文件名字)
   
   mv(move) （还可以用来改名字： mv ss.sh ssa.sh）就从ss的文件名字改成了ssa
   
   ping
   exit

**使用VIM编辑器的命令**
  
    我们可以使用vi这个命令来进入VIM编辑器去创建文件.
    such as: vi b.sh
    我们将进入vim编辑器，如果想要进行编辑，需要进入vim后输入i.进入insert模式.

![15](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/2c3afd4a-bec1-4c95-9e64-c93d495ce317)

    :w 
    保存文件但不退出 vi, 我先进入了insert模式输入了一个3*3的矩阵，然后我想进行命令。就需要先按Esc 键退出编辑模式,然后输入冒号进入命令模式.
![16](https://github.com/JoneSu1/Deep-learning-techniques-based-on-python-study-notes-and-project-records/assets/103999272/69f6eb61-9d11-49b3-b833-80a28daac238)

    :w file (:w c.sh) (这意味着)
    将修改另外保存到 file 中，不退出 vi
    :w!
    强制保存，不推出 vi
    :wq
    保存文件并退出 vi
    :wq!
    强制保存文件，并退出 vi
    q:
    不保存文件，退出 vi
    :q!
    不保存文件，强制退出 vi
    :e!
    放弃所有修改，从上次保存文件开始再编辑
  
**在vim编辑时候的插入命令**

     这个插入的意思是指在vim中写数据时候光标位置的命令操作.
     i
     在当前位置生前插入
     I
     在当前行首插入
     a
     在当前位置后插入
     A
     在当前行尾插入
     o
     在当前行之后插入一行
     O
     在当前行之前插入一行
     
**在vim编辑时候进行移动的命令**

     其中j 和k 命令是最常用的.
     h
     左移一个字符
     l
     右移一个字符，这个命令很少用，一般用 w 代
     替。
     k
     上移一个字符
     j
     下移一个字符
     以上四个命令可以配合数字使用，比如
     20j 就是
     向下移动 20 行， 5h 就是向左移动 5 个字符，在
     Vim 中，很多命令都可以配合数字使用，比如
     删除 10 个字符 10x ，在当前位置后插入 3 个！，
     3a ！！< Esc>，这里的 Esc 是必须的，否则命令不
     生效。
   
 **在vim中的拷贝和黏贴**
   
     先使用拷贝命令后再使用粘贴命令导入.
     yy 拷贝当前行
     nyy 拷贝当前后开始的 n 行，比如 2yy 拷贝当前行及其下一行。
     p 在当前光标后粘贴 如果之前使用了 yy 命令来复制一行，那么就在当前行的下一行粘贴。
     shift+p 在当前行前粘贴
     :1,10 co 20 将 1 10 行插入到第 20 行之后。
   
**在vim中的剪切命令**
   
     使用jklh是对光标所在位置的元素进行涂灰，选择.
     正常模式下按v （逐字）或 V （逐行）进入可视
     模式，然后用 jklh 命令移动即可选择某些行或
     字符，再按 d 即可剪切. (使用方向键可以达到相同效果)
     ndd
     剪切当前行之后的 n 行。利用 p 命令可以对
     剪切的内容进行粘贴
     :1,10d
     将 1 10 行剪切。利用 p 命令可将剪切后的
     内容进行粘贴。
     :1, 10 m 20
     将第 1 10 行移动到第 20 行之后。
     :1,$ co $ 将整个文件复制一份并添加到文件尾部。
   
**Linux文件管理 , 修改文件权限**

     Linux
     下文件和目录的权限区别：文件：读文件内容（ r ）、写数据到文件 w ）、作为命令执行文件（ x ）。
     目录：读包含在目录中的文件名称（r ）、写信息到目录中去（增加和删除索引点的链接）、
     搜索目录（能用该目录名称作为路径名去访问它所包含的文件和子目录）
    （1 ）有只读权限的用户不能用 cd 进入该目录，还必须有执行权限才能进入。
    （2 ）有执行权限的用户只有在知道文件名，并拥有读权利的情况下才可以访问目录下的文件。
    （3 ）必须有读和执行权限才可以 ls 列出目录清单，或使用 cd 命令进入目录。
    （4 ）有目录的写权限，可以创建、删除或修改目录下的任何文件或子目录，即使使该文件或
     子目录属于其他用户也是如此。

**文件权限的查看**
     
     ls-l xxx.xxx （xxx.xxx 是文件名）
     那么就会出现相类似的信息，主要都是这些：
     -rw rw r-- 
     共有10 位数，其中：最前面那个代表的是类型中间那三个rw 代表的是所有者（ user）
     然后那三个rw 代表的是组群（ group）
     最后那三个r 代表的是其他人（ other）
     r
     表示文件可以被读（ read
     w
     表示文件可以被写（ write
     x
     表示文件可以被执行（如果它是程序的话）
     -
     表示相应的权限还没有被授予
     
 **修改文件的权限**
 
      chmod o+w xxx.xxx
      表示给其他人授予写
      xxx.xxx 这个文件的权限
      chmod go-rw xxx.xxx
      表示删除(g表示群租，o表示其他人)
      xxx.xxx 中组群和其他人的读和写的权限
      u
      代表所有者（ user
      g
      代表所有者所在的组群（ group
      o
      代表其他人，但不是 u 和 g other
      a
      代表全部的人，也就是包括 u g 和 o
      
 ------------------------------------------
 **也可以使用数字来代替**
 
      r
      表示文件可以被读（ read ）
      w
      表示文件可以被写（ write )
      x
      表示文件可以被执行（如果它是程序的话）
      其中：rwx 也可以用数字来代替
      r -------4
      w -------2
      x -------1
      - -------0
      +  
      表示添加权限
      -
      表示删除权限
      =
      表示使之成为唯一的权限
      
 -------------------------------------
 **例子**
 
      其中括号中的（600）表示的是三个对象，u 使用者的6是说 r+w的权限数字为6， 
      第二个数字的0，表示的是群租g中没有权限，所以为0.
      第三个数字的0，表示的是其他人o中没有权限，所以为0.
      -rw --------------(600) 只有所有者才有读和写的权限
      -rw r r ----(644) 只有所有者才有读和写的权限，
      组群和其他人只有读的权限
      -rwx ------------(700) 只有所有者才有读，写，执行的权限
      -rwxr xr x (755) 只有所有者才有读，写，执行的权限，组群和其他人只有读和执行的权限
      -rwx x x (711) 只有所有者才有读，写，执行的权限，组群和其他人只有执行的权限
      -rw rw rw --(666) 每个人都有读写的权限
      -rwxrwxrwx (777) 每个人都有读写和执行的权限
