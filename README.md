# K-NN (K Nearest Neighbors) #
(Source: https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/
     http://scott.fortmann-roe.com/docs/BiasVariance.html)

## Introduction ##
-	Robust and versatile
-	Often used as benchmark for more complex algos such as ANN and SVM.
-	Used in economic forecasting, data compression, genetics

## What is KNN? ##
-	Supervised learning algorithm
-	Non-parametric: doesn’t make assumptions about the function XY or about the underlying data distribution
-	Instance-based: doesn’t explicitly learn the model; instead memorizes the training instances and only when a query to the database is made the algorithm uses the training instances to spit out an answer
-	Minimal training but huge memory cost (store a potentially large dataset) and huge computational cost (classifying each observation requires a run-down of the whole dataset)

## How does KNN work? ##
-	In classification setting, for each new observation, 

    • calculate the distance between the new observation and each observation in the entire dataset <br>
    • sort the observations basis the distances calculated<br>
    • pick the top K observations (set A) with the shortest distances (these are the K observations most similar to the new observation)<br>
    • it then estimates the conditional probability for each class, that is the fraction of points in set A with that class label
    • Finally, our new observation gets assigned to the class with the largest probability

-	Alternate way of understanding KNN is by calculating a decision boundary (or boundaries if more than 2 classes) which is used to classify new observations.

## How to select K? ##
-	K is more like a hyper-parameter that we can choose in order to get the best possible fit for the data
-	In a sense, K controls the shape of the decision boundary
-	When K is small, the classifier is “more blind” to the overall distribution. It provides a more flexible fit; the low bias and high variance. Jagged decision boundary. 
-	When K is large, there are more voters in each prediction, “more resilient to outliers”. Smoother decision boundaries. Lower variance but increased bias.



## Value of K and Bias and Variance Trade-off ##
-	At small K, jaggedness and islands are signs of variance. The location of islands and exact curve of the boundaries will radically change as new data is gathered. 
-	At large K, the transition is quite smooth so there isn’t much variance but the lack of a match to the boundary line is a sign of high bias
-	**Err(x)=Bias2+Variance+Irreducible Error**

## Managing Bias and Variance ##
### Fight Your Instincts ###

- A gut feeling many people have is that they should minimize bias even at the expense of variance. Their thinking goes that the presence of bias indicates something basically wrong with their model and algorithm. Yes, they acknowledge, variance is also bad but a model with high variance could at least predict well on average, at least it is not fundamentally wrong.

- This is mistaken logic. It is true that a high variance and low bias model can perform well in some sort of long-run average sense. However, in practice modelers are always dealing with a single realization of the data set. In these cases, long run averages are irrelevant, what is important is the performance of the model on the data you actually have and in this case bias and variance are equally important and one should not be improved at an excessive expense of the other.

### Bagging and Resampling ###

- Bagging and other resampling techniques can be used to reduce the variance in model predictions. In bagging (Bootstrap Aggregating), numerous replicates of the original data set are created using random selection with replacement. Each derivative data set is then used to construct a new model and the models are gathered together into an ensemble. To make a prediction, all of the models in the ensemble are polled and their results are averaged.

- One powerful modeling algorithm that makes good use of bagging is Random Forests. Random Forests works by training numerous decision trees each based on a different resampling of the original training data. In Random Forests the bias of the full model is equivalent to the bias of a single decision tree (which itself has high variance). By creating many of these trees, in effect a "forest", and then averaging them the variance of the final model can be greatly reduced over that of a single tree. In practice the only limitation on the size of the forest is computing time as an infinite number of trees could be trained without ever increasing bias and with a continual (if asymptotically declining) decrease in the variance.

## Understanding Over- and Under-Fitting ##
-	At its root, dealing with bias and variance is really about dealing with over- and under-fitting. Bias is reduced and variance is increased in relation to model complexity. 
-	As more and more parameters are added to a model, the complexity of the model rises, and variance becomes our primary concern while bias steadily falls. 
-	For example, as more polynomial terms are added to a linear regression, the greater the resulting model's complexity will be. In other words, bias has a negative slope in response to model complexity while variance has a positive slope.
 
-	In general, what we really care about is Overall Error, not the specific decomposition. The sweet spot for any model is the level of complexity at which the decrease in bias is equivalent to the increase in variance. 
-	If our model complexity exceeds this sweet spot, we are in effect over-fitting our model; while if our complexity falls short of the sweet spot, we are under-fitting the model. 
-	In practice, there is not an analytical way to find this location. Instead we must use an accurate measure of prediction error and explore differing levels of model complexity and then choose the complexity level that minimizes the overall error. 
-	A key to this process is the selection of an accurate error measure as often grossly inaccurate measures are used which can be deceptive. Generally resampling based measures such as cross-validation should be preferred over theoretical measures such as Aikake's Information Criteria.

## Hyperparameter Tuning (Selecting best value of K) with Cross-Validation ##
-	Best K is one that corresponds to lowest error rate
-	However, if we do repeated measurements of error rate for different values of K, we would inadvertently use the test set as training set and underestimate our true error rate. Hence test set should be touched ONLY at the very end of the pipeline.
-	An alternative and smarter approach is to use k-fold Cross Validation (holding out a subset as validation set and using it for estimating true error rate)
   1. Randomly divide the training set into k groups, or folds, of approximately equal size.
   2. The first fold is treated as a validation set, and the method is fit on the remaining k−1 folds. 
   3. The true error rate is then computed on the observations in the held-out fold. (This procedure is repeated k times; each time, a different group of observations is treated as a validation set. 
   4. The above process results in k estimates of the test error which are then averaged out. 
## Pros and Cons of KNN ##
### Pros ###
   1. Simple to understand and easy to implement. With zero to little training time, it can be a useful tool for off-the-bat analysis of some data set you are planning to run more complex algorithms on. 
   2. Furthermore, KNN works just as easily with multiclass data sets whereas other algorithms are hardcoded for the binary setting. 
   3. The non-parametric nature of KNN gives it an edge in certain settings where the data may be highly “unusual”.

### Cons ###
  1. Computationally expensive testing phase which is impractical in industry settings. Note the rigid dichotomy between KNN and the more sophisticated Neural Network which has a lengthy training phase albeit a very fast testing phase. 
  2. Furthermore, KNN can suffer from skewed class distributions. For example, if a certain class is very frequent in the training set, it will tend to dominate the majority voting of the new example (large number = more common). 
  3. Finally, the accuracy of KNN can be severely degraded with high-dimension data because there is little difference between the nearest and farthest neighbor.

## Improvements ##
- With that being said, there are many ways in which the KNN algorithm can be improved.
  1. A simple and effective way to remedy skewed class distributions is by implementing weighed voting. The class of each of the K neighbors is multiplied by a weight proportional to the inverse of the distance from that point to the given test point. This ensures that nearer neighbors contribute more to the final vote than the more distant ones.
  2. Changing the distance metric for different applications may help improve the accuracy of the algorithm. (i.e. Hamming distance for text classification)
  3. Rescaling your data makes the distance metric more meaningful. For instance, given 2 features height and weight, an observation such as x=[180,70] will clearly skew the distance metric in favor of height. One way of fixing this is by column-wise subtracting the mean and dividing by the standard deviation. Scikit-learn’s normalize() method can come in handy.
  4. Dimensionality reduction techniques like PCA should be executed prior to applying KNN and help make the distance metric more meaningful.
  5. Approximate Nearest Neighbor techniques such as using k-d trees to store the training observations can be leveraged to decrease testing time. Note however that these methods tend to perform poorly in high dimensions (20+). Try using locality sensitive hashing (LHS) for higher dimensions.
