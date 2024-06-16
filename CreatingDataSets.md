#### How to create a dataset

* Consider the following structure which can be used for both Hierarchical and Flat Classifier
![Structure for Hierarchical and Flat classifier](./images/Structure_for_flat_Hierarchical_Classifier.png)

* For Flat Classifiers we will read the images and assign them labels which are the **same as the leaf node / lowest child folder** e.g. if we read an SUV its label will be SUV
* For Hierarchical classifier if we read an image its label will be **the name of the parent folder** e.g. if we read an SUV image the label will be the parent folder which is car and reading motorbike it will be assigned the label of motorcycle

* naming convention for the folders ![Folder Naming Convention](./images/folder_naming_convention.png)
* For subfolder put both the name of the parent folder and the name of the sub folder.  It helps to keep the hierarchy in the label of the class
* Deeper Hierarchy naming convention ![Deeper Hierarchy](./images/deeper_hierarchy_subfolder.png)

#### Class Encoding for every level

* Encoding for the folders ![Cars Encoding](./images/class%20encoding.png)
* Encoding for the files ![Encoding with file IDs](./images/Adding_id_for_images.png)

#### Practical building of a data structure

* Create a **dataset** folder
* Create a **car** sub folder for vehicles with 4 wheels
* Create a **motorcycle** sub folder for vehicles with 2 wheels

**Add Sub classes to form the structure below** ![Dataset Structure](./images/folderStructure.png)

#### Data Collection Phase

* Start by keeping the parent class in the hierarchy of the class in the folder names ![Data Structure Convention](./images/datasetConvention.png)