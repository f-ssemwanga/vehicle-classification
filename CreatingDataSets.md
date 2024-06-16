#### How to create a dataset

* Consider the following structure which can be used for both Hierarchical and Flat Classifier
![Structure for Hierarchical and Flat classifier](./images/Structure_for_flat_Hierarchical_Classifier.png)

* For Flat Classifiers we will read the images and assign them labels which are the **same as the leaf node / lowest child folder** e.g. if we read an SUV its label will be SUV
* For Hierarchical classifier if we read an image its label will be **the name of the parent folder** e.g. if we read an SUV image the label will be the parent folder which is car and reading motorbike it will be assigned the label of motorcycle