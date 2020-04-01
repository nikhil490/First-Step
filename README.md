# WNS Machine Learning Challenge
XGBoost model with parameter tuning with bayesian Optimization with 5 Kfolds.

## Things Tried
1. Need to do binning of the age features.
2. Apply the mean encoding on the feature having text value and frequency more than 3.
3. Avg points * no. of comp to get the total points.
4. mean encoding or median encoding instead of label.
5. normalize or scale and then check the distribution it should be same.
6. Use the data imputation technque like(mean,median) for missing values in (education & previous_year_ratings.)
7. Add the (awards_won;KpIs_met & previous_year_rating) features,multiply the avg_training_score and no_of_training to get total training score.
8. convert education into number's where mtech>btech>other.
9. Remove the recruitment_channel that have no effect on the Target result.
10. (age - length_of_service) for gettng the joining age.
11. GradientBoostingClassifier with parameter tuning
12. RandomForestClassifier with parameter tuning
12. KNN and LinearRegression , Voting Classifier 
# Final Solution Summary
1. The missing values in the education is imputed by mode which was the "Bachelor's" & the missing value in the previous_year_rating is imputed by mode . After Correlation matrix analysis,the features like previous_year_rating,length_of_service,KPIs_met have higher correlation with the Target value.
2. The count of promotion vs no promotion is unbalanced.
3. Pca on the train set and their target values show overlapping decision boundary which can't be separated by the Linear Models.for this tye of overlapping target values Decision Trees are best. Different type of scaling on input features giving different distribution of the target values during Pca.
4. XGBoost model with parameter tuning with bayesian Optimization with 8 stratifiedKfolds.

# Newly created Features
1. avg_score  was a result of avg_mean_score divided by a mean score for the particular region and department.
2. recruitment_channel have no impact on the promotion so removed that.
3. age of joining by age - years of service
# To try
1. Apply the Pca on the input set and get the single column which summarized the input features in the 1 Dimension and used that as a     new features.helps to improve the score by 0.5 percent.
2. OOF predictions were used for finding the right threshold value.
3. kNN imputation, Random Forest imputation , MICE
# What didn't worked
1. Linear models and Neural networks gave low scores as compared to the Decision Trees.
2. mean encoding of missing values, one hot encoding of categorical values almost gave the same score.
3. blindly addition,multiplication and division of features gave low score.
4. additional features creation with Variance threshold gave same score.
5. Ensembling using Voting Classifier not gave any improvement over the single model.


# Predict Rain or Not
# First-Step
Logistic Regression with Python and Scikit-Learn and build a classifier to predict whether or not it will rain tomorrow in Australia. I train a binary classification model using Logistic Regression. I have used the Rain in Australia dataset for this project
