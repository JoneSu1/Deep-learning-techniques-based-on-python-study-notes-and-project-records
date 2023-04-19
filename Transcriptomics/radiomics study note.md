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


在进行影像组学分析时候，我们使用python作为分析工具.
我们需要用到相关的分析库有：pandas，Numpy， scipy， sciKit-learn

其中特征提取：依靠pyradiomics库进行提取的，快速且便捷.
数据预处理：
借助pandas，Numpy等库快速的完成数据预处理

数理统计和机器学习模型的设计优化：
scipy, scikit-learn库帮助我们用极少的代码实习数理统计和机器学习的建模.
![6](https://user-images.githubusercontent.com/103999272/233128526-a872b462-7e07-46e6-b71d-cdf60a82dff3.png)


同时如果不适用传统的方法进行特征提取，也可以考虑使用深度学习的方法来帮助筛选特征.
Deep learning in radiomics refers to the application of advanced artificial intelligence techniques, particularly convolutional neural networks (CNNs) and other deep learning architectures, to analyze medical images and extract quantitative features for improved diagnosis, prognosis, and treatment planning in the field of radiology.

Radiomics is a rapidly growing area of research that focuses on converting medical images into mineable high-dimensional data, enabling the extraction of valuable information from image features. By leveraging deep learning, radiomics can improve the accuracy and efficiency of medical image analysis, thereby enhancing clinical decision-making and patient outcomes.

Here's a brief introduction to the use of deep learning in radiomics:

Image Segmentation: Deep learning algorithms can automatically identify and delineate regions of interest (ROIs) in medical images, such as tumors, organs, or lesions. This allows for more accurate and consistent quantification of radiomic features, reducing the need for time-consuming manual segmentation by radiologists.

Feature Extraction: CNNs and other deep learning models can be trained to automatically extract relevant features from medical images. These features, which may include textural, morphological, and intensity-based characteristics, can be used to quantify image properties and identify patterns related to disease states or treatment response.

Prediction and Classification: Deep learning models can be used to predict clinical outcomes or classify patients based on the extracted radiomic features. For example, algorithms can be developed to predict tumor malignancy, treatment response, or patient survival, aiding in personalized medicine and tailored treatment plans.

Integration with other data sources: By combining radiomic features with other patient data, such as genetic, clinical, and demographic information, deep learning models can provide more comprehensive and accurate prognostic and predictive tools.

Enhanced visualization: Deep learning techniques can help radiologists visualize complex and subtle patterns in medical images, facilitating a better understanding of disease processes and improving diagnostic accuracy.
