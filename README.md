# Airbnb-Score-Prediction
Course project to predict probability of an Airbnb listing getting a perfect score

A surprisingly high number of Airbnbs have perfect rating scores (100/100). Airbnb owners                                    
might want to predict whether their listing will be able to achieve a perfect score.
There can be many reasons behind it:
●	Airbnb Owners want to understand the how good or bad their listing is and whether it would be able to attract customers on a daily basis
●	They would want to understand which factors contribute more to the rating and thus can improve their listing accordingly.
●	Understand customer patterns and see the period where most booking takes place and adjust prices accordingly.

 Thus it is important for the owners to understand the efficiency of their listing and thus be able to predict their score based on the airbnb features and amenities provided by them.

There are three data sets in total:
1.   airbnb_train_x_2021.csv: features for the training instances.
2.   airbnb_train_y_2021.csv: labels for the training instances (1 = has a perfect rating, 0 = does not have a perfect rating).
3.   airbnb_test_x_2021.csv: features for the test instances. 

There are a total 70 features across training and test datasets.

The primary goal for the project is to develop an accurate and an efficient model that can predict the output for the airbnb listings based on their features and amenities.
The process takes place in steps which includes:
●	Feature Engineering/Data Cleaning 
(A lot of cleaning and engineering had to be done to normalise data and create new variables so that the model gets good set of features for prediction)
●	 Building various models and seeing their performance using nested cross validation 
(Using different hyperparameters we trained several models to see their performance and then choose the best one)
●	 Prediction on Test Data and adjusting cutoff 
(Once we got the best model we got predictions for the entire test data and adjusted cutoff so that the best TPR/FPR can be obtained)
