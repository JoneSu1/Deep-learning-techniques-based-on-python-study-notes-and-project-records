What is the radiomics?
Specifically, imaging histology is the use of computers to mine a large number of quantitative image features from medical images to parse clinical information.

![1](https://user-images.githubusercontent.com/103999272/232373012-de8e1878-6626-4720-a8a6-817a3b0e6873.png)

In medical imaging, the human eye can only diagnose by visual information such as grayscale information, volume information, and morphological information.
But most important human eyes can not Accurate identify these informations. 
In contrast, the imagingomics features are high-throughput and quantified.
The more intuitive and easy-to-understand imaging features include mean, median, maximum, minimum, kurtosis, skewness, etc.

Imaging histology is mainly applied to qualitative study of tumor tissue, grading and staging, genetic analysis, etc.

![2](https://user-images.githubusercontent.com/103999272/232375316-aa9b0e13-e60f-49e7-bbce-97afa84ae1c7.png)
Assessment of time to stroke onset using an imaging histology approach.

The process of radimics:
![3](https://user-images.githubusercontent.com/103999272/232375526-e6796fe3-b17a-430d-9840-dbd5b3abc851.png)
Machine Learning Approach to Identify Stroke Within 4.5 Hours
https://doi.org/10.1161/STROKEAHA.119.027611
Usually we do not study the whole image, which is pointless and the number of acquired features is too large to make data dimensionality reduction difficult.

So, we need to look for some interest areas of this image, we want to get  useful features information. (region of interest) ROI has two form, 
1. Use lesions and tumors as study subjects. 2. Use specific functional tissues or brain regions as study subjects.  

![4](https://user-images.githubusercontent.com/103999272/232376598-89c10d0e-90ef-42e0-9925-be8f30d2733e.png)
Automated anatomical labelling atlas 3
https://doi.org/10.1016/j.neuroimage.2019.116189 
3. Image histology feature extraction
Imageomics feature extraction refers to the high-throughput extraction of a large amount of image information from medical images. 
High-throughput refers to a scientific research technique to investigate and predict the properties of substances by calculating 
a large number of properties of the system with minimum resources and maximum speed. Common image histology features include first-order statistical features,
shape features, second-order and higher-order texture features, and image features extracted by filtering with different filters. Of course, 
with the development of technological tools, some studies have extracted the depth features of images by deep learning methods to study.

![5](https://user-images.githubusercontent.com/103999272/232377374-bfb9113b-b499-4cc0-8d16-20886b613979.png)

4. PCA（影像组学特征降维）
影像组学特征的提取数目少则数十，多则数百甚至更多。 如果将所有特征都放入到机器学习模型中，会出现运算超量，出现模型过拟合等问题。
因此，在确定ROI之后，需要把获得的特征数据进行数据进行筛选和降维，仅保留缺失对实验有意义的数据。

5. 确定数据后进行模型的设计和优化
常规可以是用转录组学诊断模型构建的相关方法：例如：决策树，随机森林，逻辑回归（单因素COX，多因素COX）, 朴素贝叶斯，SVM，XGboost，K-Means等分类，回归模型。
