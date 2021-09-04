# [Breast-Cancer-Prediction](https://www.kaggle.com/jurk06/breast-cancer-classification)
To build a breast cancer classifier on an Breast Cancer Winscoin dataset that can accurately classify as benign or malignant. 
# Content
    •	Introduction
    * Who ay risk? 
    * Treatment
    •	Dataset Description
    •	Visualization
    •	Statistical Analysis
          * Correlation 
          * Multicoleniearity
          * Normality test 
          * Simpson's paradox
    •	Modelling
    •	Logistic Regression
    •	KNN
    •	SVM
    •	Decision Tree
    •	Random Forest
   **Introduction**: Breast cancer arises in the lining cells (epithelium) of the ducts (85%) or lobules (15%) in the glandular tissue of the breast. Over time, these in situ (stage 0) cancers may progress and invade the surrounding breast tissue (invasive breast cancer) then spread to the nearby lymph nodes (regional metastasis) or to other organs in the body (distant metastasis).  If a woman dies from breast cancer, it is because of widespread metastasis.
   To solve the above problem I have used many statistical techniques in order to come up with good reasoning. Initially I have checked for missing values, data pre-processing, data transformation, correlation among the features, multicollinearity, slicing the dataset, and then models were used to bring out the desire outputs. Since different classifiers were used like Logistic , Naïve bayes, KNN, Decision Tree, Random Forest, SVM and Boosting. Besides that important metrics related to the particular models were also used to check and compare which model is working very well with datasets. 
  
  
  **Who is at risk?**  Breast cancer is not a transmissible or infectious disease. There are no known viral or bacterial infections linked to the development of breast cancer. Certain factors increase the risk of breast cancer including increasing age, obesity, harmful use of alcohol, family history of breast cancer etc.
  
  
 **Dataset:-** Dataset has been collected from open source and is available at through the UW CS ftp server : ftp ftp.cs.wisc.educd math-prog/cpo-dataset/machine-learn/WDBC/.  Besides that it is also available at Kaggle. 
 ![features name](https://user-images.githubusercontent.com/22790745/132085238-321f22b5-4256-4444-bb96-f7223b6464bc.png)
 It contains 32 features and 532 rows. 
 
 
 **Explanatory and Response variable** As we can see that there 
Features	| variables Type
------------|---------------
diagnosis	 |Response
rest 30 features	|Expalanatory


**Response Variables:-**   :-  It can bee see that ![pie chart](https://user-images.githubusercontent.com/22790745/132085295-c0da7ea9-174c-4e91-9587-bef839a76bd2.png)
From the figure it can be observed that there “B” is 62% while “M” is almost 38%. We can see a kind of data imbalanced. Though in our analysis I have considered two cases


**Missing Values:-** After carefully analysing the dataset we have seen that it is perfect dataset which contains no null values. Hence our work became easy as no imputation has been performed here. 

**Pearson’s Correlation Coefficient:**- To find out the relationship between the response variable and each explanatory variable
![formula](https://user-images.githubusercontent.com/22790745/132085352-1a833ac6-4466-4237-acff-ad0cfa23db75.png).

* Case-1: when r=-1, it shows that both the variables are negatively correlated. With the increase of the one variable other will decrease.

* Case-2: when r=0, it shows that both the features are independent.

* Case-3: When r=1, it shows that both the features are highly related.

![correlation Matrix](https://user-images.githubusercontent.com/22790745/132085412-28bfe4ae-109a-4f75-8d9c-39473579e204.png)


**Spearman’s Correlation Coefficient** :- it is same as with Pearson’s correlation coefficient.


**Multi-colinearity**: As we can see that there many features in the dataset i.e. there are 32 features including the response variable. There is high chance that there might be multicolinearlity among the features. To detect the multicolinearlity we have multiple options.
   •	Variance inflation factor (VIF) measures how much the behaviour (variance) of an independent variable is influenced, or inflated, by its interaction/correlation with the other independent variables. 
   •	1 = not correlate.
   •	Between 1 and 5 = moderately correlate.
   •	Greater than 5 = highly correlate.
   *	VIF> 10 is used and remove the feature which has that high VIF.

![VIF-calculation](https://user-images.githubusercontent.com/22790745/132085933-00bcd0ea-fdba-4b4f-80e9-aefc87da864d.png)


**Normality-Test** :- •	In our case VIF is not working well. So I have chosen an alternative option.  We have to perform some statistical analysis. We have to calculate the p-value for each feature. We remove those features which have higher p-value. •	Next is normality test is performed for each of the variables. Those features which do not follow the Gaussian distribution will be not used for the modelling. To perform the normality test, we can use QQ plot, distribution plot and histogram. Beside that some KS test and chi-square goodness of fit are also performed

**Q-Q plot** and **Shapiro-Wilk Test** 

**Simpson's Paradox**:- We have done slicing of the dataset in order to observe a particular trends in dataset or some kind of the distribution which might be not avilable with the whole dataset. It is called as simpsons's paradox.
In my case I didn't find such paradox.

# Moddel 
* As we can see that now the time to modesl
Backward stepwise
	Backward stepwise selection (or backward elimination) is a variable selection method which:
	Begins with a model that contains all variables under consideration (called the Full Model)
	Then starts removing the least significant variables one after the other
	Until a pre-specified stopping rule is reached or until no variable is left in the model
	From Full model we eliminate the least important features.
Determine the least significant variable to remove at each step
The least significant variable is a variable that:
	Has the highest p-value in the model, or
	Its elimination from the model causes the lowest drop in R2, or
	Its elimination from the model causes the lowest increase in RSS (Residuals Sum of Squares) compared to other predictors. # Choose a stopping rule The stopping rule is satisfied when all remaining variables in the model have a p-value smaller than some pre-specified threshold. When we reach this state, backward elimination will terminate and return the current step’s model. # Where backward stepwise is better
	Starting with the full model has the advantage of considering the effects of all variables simultaneously.
	This is especially important in case of collinearity (when variables in a model are correlated which each other) because backward stepwise may be forced to keep them all in the model unlike forward selection where none of them might be entered
	Unless the number of candidate variables > sample size (or number of events), use a backward stepwise approach.






Metrics for Classification:- The metrics are used for obtaining the accuracy of the model is 
	
    Accuracy Score- In classification accuracy is for a given points how many points are correctly classified. It is good metrics for the balanced class but it will not give good values for the case of imbalanced dataset.
	
    Confusion Metrics- It is tool to find out the accuracy of the model. It is matrix of the true class and predicted class. ![image](https://user-images.githubusercontent.com/22790745/132086220-1b7ef560-bb5a-494f-a2d1-e5301c273727.png)

  
	Precision:- It is the ratio of True Positive/(True Positive + False Positive). It is one of the good metrics for imbalanced dataset. As we want the Precision to be 1 i.e. when FP=0.
	
    Recall:- Ratio of True Positive/(True Positive + False negative) . It is one of the good metrics for imbalanced dataset. As we want the Recall to be 1 i.e. when FN=0. Since in our case we want recall to be high. We want  to reduce the Type-I error 
	
    F-1 score- It is the harmonic mean of Precision and Recall. 
	
    
    ROC-AUC score:  It is stand for Receiver Operating Characteristics curve.  Drawing the curve between the True Positive Rate and False Positive rate. For each threshold value of the probabilities, we get different value of FPR and TPR; according to this we plot the curve. 
	
    
    Precision Recall curve: It is plot between the precision and recall. It is good measure for the imbalanced dataset. A precision-recall curve is a plot of the precision (y-axis) and the recall (x-axis) for different thresholds, much like the ROC curve. Dumb model gives an area of equal to 0.5. 
 




**Logistic Regression**: It is basically a classification algorithm used to classify the binary classification mainly. Since the algorithm is developed in such a way that the outcome will predict the class on the basis of the probability. Since the class are classified on the basis of the threshold probability. 
The formula is like log of odd’s ratio is linearly related. This is function is called as Sigmoid Function.  We know the formula. 

Loss Function: The logistic loss function is calculated as
w*=argmin∑▒〖ln⁡(1+e^(-y*w^tx ))〗
                Where y*w^tx  is the sign distance from the plane which is optimum.

Regularization: It is a penalty which is used to reduce the loss function and helps to reduce the over fitting. As we know that sometime the machine learning model which perform very well on training dataset but on test dataset those models become over fitting. Over fitting is due to the consideration of the noise dataset. This is why we use regularization technique. As it helps to reduce the over fitting of the model. 
w*=argmin∑▒〖ln⁡(1+e^(-y*w^tx ) )+Regularization〗

Two type of regularization are there

**Ridge Regression**: Ridge regression is one of the types of linear regression in which we introduce a small amount of bias, known as Ridge regression penalty so that we can get better long-term predictions.
	In Statistics, it is known as the L-2 norm.
	In this technique, the cost function is altered by adding the penalty term (shrinkage term),
	which multiplies the lambda with the squared weight of each individual feature
	Therefore, the optimization function(cost function) becomes: 
	w*=argmin∑▒〖ln⁡(1+e^(-y*w^tx ) )+\labda ∑▒β^2 〗
    
# Usage of Ridge Regression:
	When we have the independent variables which are having high collinearity (problem of multicollinearity) between them, at that time general linear or polynomial regression will fail so to solve such problems, Ridge regression can be used.
	If we have more parameters than the samples, then Ridge regression helps to solve the problems.
    
# Limitation of Ridge Regression 
	Not helps in Feature Selection: It decreases the complexity of a model but does not reduce the number of independent variables since it never leads to a coefficient being zero rather only minimizes it
	Model Interpretability: Its disadvantage is model interpretability since it will shrink the coefficients for least important predictors, very close to zero but it will never make them exactly zero. In other words, the final model will include all the independent variables, also known as predictors.
    
# Lasso Regression: 
	Lasso regression is another variant of the regularization technique used to reduce the complexity of the model. It stands for Least Absolute and Selection Operator.
	It is similar to the Ridge Regression except that the penalty term includes the absolute weights instead of a square of weights.
	w*=argmin∑▒〖ln⁡(1+e^(-y*w^tx ) )+\labda ∑▒β〗
	In statistics, it is known as the L-1 norm.
	In this technique, the L1 penalty has the eﬀect of forcing some of the coeﬃcient estimates to be exactly equal to zero which means there is a complete removal of some of the features for model evaluation when the tuning parameter λ is suﬃciently large. 
	Therefore, the lasso method also performs Feature selection and is said to yield sparse models.
    
 # Limitation of Lasso Regression:
	Problems with some types of Dataset: If the number of predictors is greater than the number of data points, Lasso will pick at most n predictors as non-zero, even if all predictors are relevant.
	Multicollinearity Problem: If there are two or more highly collinear variables then LASSO regression selects one of them randomly which is not good for the interpretation of our model?
    
# Key Differences between Ridge and Lasso Regression
	In Ridge Regression only overfit is overcome but in Lasso regression both Overfit and feature selection problem are resolved. 
	Lasso Regression tends to make coefficients to absolute zero whereas Ridge regression never sets the value of coefficient to absolute zero.
Importante points about λ:
	λ is the tuning parameter used in regularization that decides how much we want to penalize the flexibility of our model i.e, controls the impact on bias and variance.
	λ value incrasses , variance decreases and hence overfiting is  avoided . But after further increase in value causes, bias to increase and undrefitting happens.
	When λ = 0, the penalty term has no eﬀect, the equation becomes the cost function of the linear regression model. Hence, for the minimum value of λ i.e, λ=0, the model will resemble the linear regression model. So, the estimates produced by ridge regression will be equal to least squares
	However, as λ→∞ (tends to infinity), the impact of the shrinkage penalty increases, and the ridge regression coeﬃcient estimates will approach zero.
Feature Importance:-  It is one of the technique to more about that which feature is most important. To get the features importance we have to find out the coefficients of each of the features without and with standardization. As we can see that standardization affects the features importance. 
In case of logistic regression “Radius se” was most important feature, I mean it had high and positive coefficient. Since Logistic regression deals with coefficients of the models, while in other models directly compute the important feature values.
Feature after Standaization: As I have observed that feature importance greatly impacted by the standardization of the scales. I believe standardization is must need before finding out the feature important.  


**KNN**:- It is distance based non-parametric approach for the classification. The mail. KNN can be used for both classification and regression predictive problems. However, it is more widely used in classification problems in the industry.
 
Algorithm It is simple and elegant algorithm. To classify the point we take few nearest neighbours of those points. The distance of the point from those points are calculated and then majority votes are counted. This is the way to classify the points. This is how the algorithm works. 
How to choose right “K”: 
	Since K is hyper parameter. The value of K decides whether the curve is going to be overfit or underfit.  When K=1 the decision boundary is overfit and K= large the decision boundary is underfit. 
	At K=1 the training error is zero but still this value of K is not taken into consideration.
	To find the optimum K value we perform the Cross validation and we calculate the cross validation error. All I need is to plot the curve Cross validation Error with respective K. The K which gives the least CV error , That K is optimum K. 
Variance Bias Trade Off: 
	At K=1 we see that it is overfitting the decision boundary and the variance is very high and bias is low
	As K increase the overfitting is reduced and variance is getting decreased but bias is increased. 
    
    
# Decision Tree
Decision Tree:  It is one the finest algorithm for classification in machine learning.  It is nested if else tree structure. In statistics its algorithm works in a way that a number of hyperplanes (Axis parallel hyperplanes) are drawn in order to classify the different response classes. In decision tree we see Root Node, Terminal Nodes, Decision Nodes, branches etc.
	Root Nodes- From where the decision tree starts, it is the first nodes where splitting occurs. 
	Decision Nodes- Nodes other than Root nodes, here splitting takes places.
	Terminal Nodes:- Here is the final nodes when no other splitting occurs.
Splitting Criteria Of Nodes: As we know that there are specific criteria on the basis of this the splitting of the nodes occurs. 
Case-I:- Entropy: This is the first criteria on the basis of this splitting of the nodes takes place. We can think about the Entropy of a dataset in terms of the probability distribution of observations in the dataset belonging to one class or another, e.g. two classes in the case of a binary classification dataset.
H(s)=-∑▒〖P(x)ln⁡(P(x))〗 
Where P(x) is the probability of obtaining the different classes.
	Since we see that Entropy works well when the probability of obtaining the positive and negative class are same i.e. P(x1)=0.5, P(x2)=0.5, Entropy is almost 1
	When P(x1)=1 & P(x2)=0, Entropy is zero i.e. H(s)=0
	When P(x1)=0.9 & P(x2)=0.01 Entropy is very less i.e. H(s)=0.02
	Entropy quantifies how much information there is in a random variable, or more specifically its probability distribution. 
	A skewed distribution has low entropy, whereas a distribution where events have equal probability has larger entropy.
Information Gain: 
	Information Gain, or IG for short, measures the reduction in entropy or surprise by splitting a dataset according to a given value of a random variable.
	A larger information gain suggests a lower entropy group or groups of samples, and hence less surprise.
	You might recall that information quantifies how surprising an event is in bits. Lower probability events have more information; higher probability events have less information.
	In information theory, we like to describe the “surprise” of an event. Low probability events are more surprising therefore has a larger amount of information.
	Whereas probability distributions where the events are equally likely are more surprising and have larger entropy
	Skewed Probability Distribution (unsurprising): Low entropy.
	Balanced Probability Distribution (surprising): High entropy.
	Information gain provides a way to use entropy to calculate how a change to the dataset impacts the purity of the dataset, e.g. the distribution of classes. Smaller entropy suggests more purity or less surprise.
 

Gini Impurity:- It is the case 2 of splitting criteria. The formula for the IG is given by 
I_G=1-[P^2 (x1)+P^2 (x2)]
	The formula is quite same as Entropy.
	Since it is computation easy as we are not using the logarithmic terms
	 the Gini Impurity of a dataset is a number between 0-0.5, which indicates the likelihood of new, random data being misclassified if it were given a random class label according to the class distribution in the dataset.
	This calculation would measure the impurity of the split, and the feature with the lowest impurity would determine the best feature for splitting the current node. 
	This process would continue for each subsequent node using the remaining features.
	The features with least Gini impurity are selected as the root node.
	An attribute with the smallest Gini Impurity is selected for splitting the node.
	In order to obtain information gain for an attribute, the weighted impurities of the branches is subtracted from the original impurity. The best split can also be chosen by maximizing the Gini gain. Gini gain is calculated as follows:  G(a)=G(A-parent)-G(A-child)
This is how the decision tree splitting works
Hyperparameters: The hyper-parameters in decision tree is depth of the tree. 
	We keep decision tree as shallow due to which the overfitting can be avoided
	When the depth of the decision tree is deep , overfitting occurs.























