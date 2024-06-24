
# Vehicle Type Classification

The aim of the project is to build an AI project which classifies vehicles.  We need to build a commercial-level surveillance system that counts the number of cars and motorcycles

* The first task is to classify motorcycles and cars.
![Vehicle Classification](./vehicleClassification.png)

#### How to build the classifier?

* Consider a hierarchical classifier Tree ![Hierarchical Classifier Tree](./images/three_classifiers.png)

* Chosen Hierarchical Classes ![Resulting Hierarchical classifiers](./images/vehicleClassification.png)

#### Limitations of Hierarchical Classifiers

* A deeper hierarchy would require more classifiers to build and this might not end up not being very practical
* The advantage is that every classifier focuses on a small number of classes to recognise, making it easier to train as there are reasonable requirements in terms of memory and computation.  Usually, these classifiers can be trained on one GPU 
* Requires as many classifiers as there are parent-to-child relationships in the hierarchy and you will need several trainings to develop each model.  During prediction, you will need to go through several models in the hierarchy to get the classification of a single object.

For example, a classification of a car would need to go through the first classifier to identify it as either a car or a motorcycle. If it is a car it would have to go through to  the second classifier to be recognised as either Bus, SUV or Sedan.  This will increase the prediction time.

If an object is misclassified at the parent level, all subsequent classifications would go wrong too at the next level

#### The second approach is to develop a flat classifier

* Flat file Classifier ![Flat File Classifier](./images/flatClassifier.png)
* All vehicle types are considered at the same level, with one classifier and no parent-to-child relationship.
* From a design perspective this might be easy to build but might be very greedy in terms of memory - e.g. if we have 100 classes the output would be a **softmax** function with 100 output values, which makes the classifier quite complex.
* In real-world problems you will need a supercomputer to handle this and the number of images will be much larger.
* In a hierarchical classifier each classifier would use a subset of images

#### Summary

* Decision will depend on the size of the dataset
* Computational resources available
* Problem Complexity

#### Increased problem complexity

* Addition classes in the MotorBike category ![Motor Bike](./images/increased%20complexity2.png)
* Additional classes in the Bicycle category ![Bicycle](./images/increased%20complexity1.png)

