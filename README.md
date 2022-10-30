# Logistic Regression[](https://xuecxehx.labs.coursera.org/notebooks/C1_W3_Logistic_Regression.ipynb#Logistic-Regression)

In this exercise, you will implement logistic regression and apply it to two different datasets.

# Outline[](https://xuecxehx.labs.coursera.org/notebooks/C1_W3_Logistic_Regression.ipynb#Outline)

-   1 - Packages
-   2 - Logistic Regression
    -   2.1 Problem Statement
    -   2.2 Loading and visualizing the data
    -   2.3 Sigmoid function
    -   2.4 Cost function for logistic regression
    -   2.5 Gradient for logistic regression
    -   2.6 Learning parameters using gradient descent
    -   2.7 Plotting the decision boundary
    -   2.8 Evaluating logistic regression
-   3 - Regularized Logistic Regression
    -   3.1 Problem Statement
    -   3.2 Loading and visualizing the data
    -   3.3 Feature mapping
    -   3.4 Cost function for regularized logistic regression
    -   3.5 Gradient for regularized logistic regression
    -   3.6 Learning parameters using gradient descent
    -   3.7 Plotting the decision boundary
    -   3.8 Evaluating regularized logistic regression model
## 1 - Packages

First, let's run the cell below to import all the packages that you will need during this assignment.

-   [numpy](https://xuecxehx.labs.coursera.org/notebooks/www.numpy.org)  is the fundamental package for scientific computing with Python.
-   [matplotlib](http://matplotlib.org/)  is a famous library to plot graphs in Python.
-   `utils.py`  contains helper functions for this assignment. You do not need to modify code in this file.
## 2 - Logistic Regression

In this part of the exercise, you will build a logistic regression model to predict whether a student gets admitted into a university.

### 2.1 Problem Statement

Suppose that you are the administrator of a university department and you want to determine each applicant’s chance of admission based on their results on two exams.

-   You have historical data from previous applicants that you can use as a training set for logistic regression.
-   For each training example, you have the applicant’s scores on two exams and the admissions decision.
-   Your task is to build a classification model that estimates an applicant’s probability of admission based on the scores from those two exams.

### 2.2 Loading and visualizing the data

You will start by loading the dataset for this task.

-   The  `load_dataset()`  function shown below loads the data into variables  `X_train`  and  `y_train`
    -   `X_train`  contains exam scores on two exams for a student
    -   `y_train`  is the admission decision
        -   `y_train = 1`  if the student was admitted
        -   `y_train = 0`  if the student was not admitted
    -   Both  `X_train`  and  `y_train`  are numpy arrays.
### 2.3 Sigmoid function[](https://xuecxehx.labs.coursera.org/notebooks/C1_W3_Logistic_Regression.ipynb#2.3--Sigmoid-function)

Recall that for logistic regression, the model is represented as

𝑓𝐰,𝑏(𝑥)=𝑔(𝐰⋅𝐱+𝑏)fw,b(x)=g(w⋅x+b)

where function  𝑔g  is the sigmoid function. The sigmoid function is defined as:

𝑔(𝑧)=11+𝑒−𝑧g(z)=11+e−z

Let's implement the sigmoid function first, so it can be used by the rest of this assignment.

### Exercise 1

Please complete the  `sigmoid`  function to calculate

𝑔(𝑧)=11+𝑒−𝑧g(z)=11+e−z

Note that

-   `z`  is not always a single number, but can also be an array of numbers.
-   If the input is an array of numbers, we'd like to apply the sigmoid function to each value in the input array.

If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.
### 2.4 Cost function for logistic regression

In this section, you will implement the cost function for logistic regression.

### Exercise 2

Please complete the  `compute_cost`  function using the equations below.

Recall that for logistic regression, the cost function is of the form

𝐽(𝐰,𝑏)=1𝑚∑𝑖=0𝑚−1[𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖))](1)(1)J(w,b)=1m∑i=0m−1[loss(fw,b(x(i)),y(i))]

where

-   m is the number of training examples in the dataset

-   𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖))loss(fw,b(x(i)),y(i))  is the cost for a single data point, which is -
    
    𝑙𝑜𝑠𝑠(𝑓𝐰,𝑏(𝐱(𝑖)),𝑦(𝑖))=(−𝑦(𝑖)log(𝑓𝐰,𝑏(𝐱(𝑖)))−(1−𝑦(𝑖))log(1−𝑓𝐰,𝑏(𝐱(𝑖)))(2)(2)loss(fw,b(x(i)),y(i))=(−y(i)log⁡(fw,b(x(i)))−(1−y(i))log⁡(1−fw,b(x(i)))
    

-   𝑓𝐰,𝑏(𝐱(𝑖))fw,b(x(i))  is the model's prediction, while  𝑦(𝑖)y(i), which is the actual label
    
-   𝑓𝐰,𝑏(𝐱(𝑖))=𝑔(𝐰⋅𝐱(𝐢)+𝑏)fw,b(x(i))=g(w⋅x(i)+b)  where function  𝑔g  is the sigmoid function.
    
    -   It might be helpful to first calculate an intermediate variable  𝑧𝐰,𝑏(𝐱(𝑖))=𝐰⋅𝐱(𝐢)+𝑏=𝑤0𝑥(𝑖)0+...+𝑤𝑛−1𝑥(𝑖)𝑛−1+𝑏zw,b(x(i))=w⋅x(i)+b=w0x0(i)+...+wn−1xn−1(i)+b  where  𝑛n  is the number of features, before calculating  𝑓𝐰,𝑏(𝐱(𝑖))=𝑔(𝑧𝐰,𝑏(𝐱(𝑖)))fw,b(x(i))=g(zw,b(x(i)))

Note:

-   As you are doing this, remember that the variables  `X_train`  and  `y_train`  are not scalar values but matrices of shape (𝑚,𝑛m,n) and (𝑚𝑚,1) respectively, where  𝑛𝑛  is the number of features and  𝑚𝑚  is the number of training examples.
-   You can use the sigmoid function that you implemented above for this part.

If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.
### 2.5 Gradient for logistic regression

In this section, you will implement the gradient for logistic regression.

Recall that the gradient descent algorithm is:

repeat until convergence:{𝑏:=𝑏−𝛼∂𝐽(𝐰,𝑏)∂𝑏𝑤𝑗:=𝑤𝑗−𝛼∂𝐽(𝐰,𝑏)∂𝑤𝑗}for j := 0..n-1(1)repeat until convergence:{b:=b−α∂J(w,b)∂b(1)wj:=wj−α∂J(w,b)∂wjfor j := 0..n-1}

where, parameters  𝑏b,  𝑤𝑗wj  are all updated simultaniously

### Exercise 3

Please complete the  `compute_gradient`  function to compute  ∂𝐽(𝐰,𝑏)∂𝑤∂J(w,b)∂w,  ∂𝐽(𝐰,𝑏)∂𝑏∂J(w,b)∂b  from equations (2) and (3) below.

∂𝐽(𝐰,𝑏)∂𝑏=1𝑚∑𝑖=0𝑚−1(𝑓𝐰,𝑏(𝐱(𝑖))−𝐲(𝑖))(2)(2)∂J(w,b)∂b=1m∑i=0m−1(fw,b(x(i))−y(i))

∂𝐽(𝐰,𝑏)∂𝑤𝑗=1𝑚∑𝑖=0𝑚−1(𝑓𝐰,𝑏(𝐱(𝑖))−𝐲(𝑖))𝑥(𝑖)𝑗(3)(3)∂J(w,b)∂wj=1m∑i=0m−1(fw,b(x(i))−y(i))xj(i)

-   m is the number of training examples in the dataset

-   𝑓𝐰,𝑏(𝑥(𝑖))fw,b(x(i))  is the model's prediction, while  𝑦(𝑖)y(i)  is the actual label

-   **Note**: While this gradient looks identical to the linear regression gradient, the formula is actually different because linear and logistic regression have different definitions of  𝑓𝐰,𝑏(𝑥)fw,b(x).

As before, you can use the sigmoid function that you implemented above and if you get stuck, you can check out the hints presented after the cell below to help you with the implementation.
### 2.6 Learning parameters using gradient descent

Similar to the previous assignment, you will now find the optimal parameters of a logistic regression model by using gradient descent.

-   You don't need to implement anything for this part. Simply run the cells below.
    
-   A good way to verify that gradient descent is working correctly is to look at the value of  𝐽(𝐰,𝑏)J(w,b)  and check that it is decreasing with each step.
    
-   Assuming you have implemented the gradient and computed the cost correctly, your value of  𝐽(𝐰,𝑏)J(w,b)  should never increase, and should converge to a steady value by the end of the algorithm.
### 2.7 Plotting the decision boundary

We will now use the final parameters from gradient descent to plot the linear fit. If you implemented the previous parts correctly, you should see the following plot:  
![](https://xuecxehx.labs.coursera.org/notebooks/images/figure%202.png)

We will use a helper function in the  `utils.py`  file to create this plot.
### 2.8 Evaluating logistic regression

We can evaluate the quality of the parameters we have found by seeing how well the learned model predicts on our training set.

You will implement the  `predict`  function below to do this.

### Exercise 4

Please complete the  `predict`  function to produce  `1`  or  `0`  predictions given a dataset and a learned parameter vector  𝑤w  and  𝑏b.

-   First you need to compute the prediction from the model  𝑓(𝑥(𝑖))=𝑔(𝑤⋅𝑥(𝑖)+𝑏)f(x(i))=g(w⋅x(i)+b)  for every example
    
    -   You've implemented this before in the parts above
-   We interpret the output of the model (𝑓(𝑥(𝑖))f(x(i))) as the probability that  𝑦(𝑖)=1y(i)=1  given  𝑥(𝑖)x(i)  and parameterized by  𝑤w.
    
-   Therefore, to get a final prediction (𝑦(𝑖)=0y(i)=0  or  𝑦(𝑖)=1y(i)=1) from the logistic regression model, you can use the following heuristic -
    
    if  𝑓(𝑥(𝑖))>=0.5f(x(i))>=0.5, predict  𝑦(𝑖)=1y(i)=1
    
    if  𝑓(𝑥(𝑖))<0.5f(x(i))<0.5, predict  𝑦(𝑖)=0y(i)=0
    

If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.
## 3 - Regularized Logistic Regression

In this part of the exercise, you will implement regularized logistic regression to predict whether microchips from a fabrication plant passes quality assurance (QA). During QA, each microchip goes through various tests to ensure it is functioning correctly.

### 3.1 Problem Statement

Suppose you are the product manager of the factory and you have the test results for some microchips on two different tests.

-   From these two tests, you would like to determine whether the microchips should be accepted or rejected.
-   To help you make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model.

### 3.2 Loading and visualizing the data

Similar to previous parts of this exercise, let's start by loading the dataset for this task and visualizing it.

-   The  `load_dataset()`  function shown below loads the data into variables  `X_train`  and  `y_train`
    -   `X_train`  contains the test results for the microchips from two tests
    -   `y_train`  contains the results of the QA
        -   `y_train = 1`  if the microchip was accepted
        -   `y_train = 0`  if the microchip was rejected
    -   Both  `X_train`  and  `y_train`  are numpy arrays.
### 3.3 Feature mapping

One way to fit the data better is to create more features from each data point. In the provided function  `map_feature`, we will map the features into all polynomial terms of  𝑥1x1  and  𝑥2x2  up to the sixth power.


As a result of this mapping, our vector of two features (the scores on two QA tests) has been transformed into a 27-dimensional vector.

-   A logistic regression classifier trained on this higher-dimension feature vector will have a more complex decision boundary and will be nonlinear when drawn in our 2-dimensional plot.
-   We have provided the  `map_feature`  function for you in utils.py.
- ### 3.4 Cost function for regularized logistic regression

In this part, you will implement the cost function for regularized logistic regression.

Recall that for regularized logistic regression, the cost function is of the form

𝐽(𝐰,𝑏)=1𝑚∑𝑖=0𝑚−1[−𝑦(𝑖)log(𝑓𝐰,𝑏(𝐱(𝑖)))−(1−𝑦(𝑖))log(1−𝑓𝐰,𝑏(𝐱(𝑖)))]+𝜆2𝑚∑𝑗=0𝑛−1𝑤2𝑗J(w,b)=1m∑i=0m−1[−y(i)log⁡(fw,b(x(i)))−(1−y(i))log⁡(1−fw,b(x(i)))]+λ2m∑j=0n−1wj2

Compare this to the cost function without regularization (which you implemented above), which is of the form

𝐽(𝐰.𝑏)=1𝑚∑𝑖=0𝑚−1[(−𝑦(𝑖)log(𝑓𝐰,𝑏(𝐱(𝑖)))−(1−𝑦(𝑖))log(1−𝑓𝐰,𝑏(𝐱(𝑖)))]J(w.b)=1m∑i=0m−1[(−y(i)log⁡(fw,b(x(i)))−(1−y(i))log⁡(1−fw,b(x(i)))]

The difference is the regularization term, which is

𝜆2𝑚∑𝑗=0𝑛−1𝑤2𝑗λ2m∑j=0n−1wj2

Note that the  𝑏b  parameter is not regularized.

### Exercise 5

Please complete the  `compute_cost_reg`  function below to calculate the following term for each element in  𝑤w

𝜆2𝑚∑𝑗=0𝑛−1𝑤2𝑗λ2m∑j=0n−1wj2

The starter code then adds this to the cost without regularization (which you computed above in  `compute_cost`) to calculate the cost with regulatization.

If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.
### 3.5 Gradient for regularized logistic regression

In this section, you will implement the gradient for regularized logistic regression.

The gradient of the regularized cost function has two components. The first,  ∂𝐽(𝐰,𝑏)∂𝑏∂J(w,b)∂b  is a scalar, the other is a vector with the same shape as the parameters  𝐰w, where the  𝑗thjth  element is defined as follows:

∂𝐽(𝐰,𝑏)∂𝑏=1𝑚∑𝑖=0𝑚−1(𝑓𝐰,𝑏(𝐱(𝑖))−𝑦(𝑖))∂J(w,b)∂b=1m∑i=0m−1(fw,b(x(i))−y(i))

∂𝐽(𝐰,𝑏)∂𝑤𝑗=(1𝑚∑𝑖=0𝑚−1(𝑓𝐰,𝑏(𝐱(𝑖))−𝑦(𝑖))𝑥(𝑖)𝑗)+𝜆𝑚𝑤𝑗for 𝑗=0...(𝑛−1)∂J(w,b)∂wj=(1m∑i=0m−1(fw,b(x(i))−y(i))xj(i))+λmwjfor j=0...(n−1)

Compare this to the gradient of the cost function without regularization (which you implemented above), which is of the form

∂𝐽(𝐰,𝑏)∂𝑏=1𝑚∑𝑖=0𝑚−1(𝑓𝐰,𝑏(𝐱(𝑖))−𝐲(𝑖))(2)(2)∂J(w,b)∂b=1m∑i=0m−1(fw,b(x(i))−y(i))

∂𝐽(𝐰,𝑏)∂𝑤𝑗=1𝑚∑𝑖=0𝑚−1(𝑓𝐰,𝑏(𝐱(𝑖))−𝐲(𝑖))𝑥(𝑖)𝑗(3)(3)∂J(w,b)∂wj=1m∑i=0m−1(fw,b(x(i))−y(i))xj(i)

As you can see,∂𝐽(𝐰,𝑏)∂𝑏∂J(w,b)∂b  is the same, the difference is the following term in  ∂𝐽(𝐰,𝑏)∂𝑤∂J(w,b)∂w, which is

𝜆𝑚𝑤𝑗for 𝑗=0...(𝑛−1)λmwjfor j=0...(n−1)

### Exercise 6

Please complete the  `compute_gradient_reg`  function below to modify the code below to calculate the following term

𝜆𝑚𝑤𝑗for 𝑗=0...(𝑛−1)λmwjfor j=0...(n−1)

The starter code will add this term to the  ∂𝐽(𝐰,𝑏)∂𝑤∂J(w,b)∂w  returned from  `compute_gradient`  above to get the gradient for the regularized cost function.

If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.
### 3.6 Learning parameters using gradient descent

Similar to the previous parts, you will use your gradient descent function implemented above to learn the optimal parameters  𝑤w,𝑏b.

-   If you have completed the cost and gradient for regularized logistic regression correctly, you should be able to step through the next cell to learn the parameters  𝑤w.
-   After training our parameters, we will use it to plot the decision boundary.

**Note**

The code block below takes quite a while to run, especially with a non-vectorized version. You can reduce the  `iterations`  to test your implementation and iterate faster. If you have time, run for 100,000 iterations to see better results.
### 3.7 Plotting the decision boundary

To help you visualize the model learned by this classifier, we will use our  `plot_decision_boundary`  function which plots the (non-linear) decision boundary that separates the positive and negative examples.

-   In the function, we plotted the non-linear decision boundary by computing the classifier’s predictions on an evenly spaced grid and then drew a contour plot of where the predictions change from y = 0 to y = 1.
    
-   After learning the parameters  𝑤w,𝑏b, the next step is to plot a decision boundary similar to Figure 4.
    

![](https://xuecxehx.labs.coursera.org/notebooks/images/figure%204.png)
### 3.8 Evaluating regularized logistic regression model

You will use the  `predict`  function that you implemented above to calculate the accuracy of the regulaized logistic regression model on the training set
