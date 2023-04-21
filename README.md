# Module-11-Overfitting-and-Regularization

## Objectives

At the end of the module you will be able to
- Understand why overfitting occurs and why it is a problem
- Understand what regularization is and how it can reduce overfitting to noise
- Understand how lasso and ridge regression work and the difference between them
- Apply lasso and ridge regression
- Apply cross-validation to determine the optimal hyperparameter values for lasso and ridge regression

## Materials
- Read and go through './docs/overfittingAndRegularization.mlx'.

## Assignment
Complete the following assignment in './code/restFC.m':
1. Begin by loading the data (rest_fMRI.mat), and preproccess by z-scoring the rest data (run1 and run2) within each region/subject.  The data arrays have the structure [time x region x subject].
2. Calculate two functional connectivity (FC) matrices for each subject for the run1 data: one using multiple regression and the other using lasso/elastic net.  To construct an FC matrix using regression, start by preallocating an [nrRegions x nrRegions] matrix for each subject.  Then you must loop through every region (also referred to as a node), giving each a turn as “target” region.  Fit the regression model with y being the target node’s timeseries and X being the timeseries of all nodes but the target (source nodes).  Then, in the matrix row that corresponds with the target node, fill in the resulting beta coefficients.  There should be one beta for every source node.  Skip the column corresponding to the target node (diagonal), since there is not a coefficient for itself.  The diagonal should remain zeros or NaNs when the matrix is entirely filled.
3. Perform cross validation to determine the optimal lambda AND alpha values for the lasso/elastic net models.  For the first subject, find the single lambda and alpha values out of a range that will provide the best MSE when averaged across regions.  What are the values?  Make a surface/mesh plot of region-averaged MSE at each alpha-lambda combination.  Then use that one subject’s lambda and alpha for all models.  Or, if not too computationally expensive, find the optimal hyperparameters for every subject and fit each subject’s lasso/elastic net models using their own personalized lambda and alpha.
4. Once FC matrices have been computed for multiple regression and lasso/elastic net, use imagesc() to plot the two matrices for the first subject.  
5. Then test the generalizability of the multiple regression and lasso FC coefficients by their ability to predict held-out data (run2).  For each region x subject, calculate R^2 between predicted and actual run2 data.  What is the average R^2 for multiple regression FC and for lasso/elastic net FC?
6. In the provided data, half of the subjects are younger adults (36-40 years old) while half are older adults (>65 years old; see subjGroup).  Perform an independent samples t-test between younger and older adults to test for a group difference in every connection, separately using multiple regression and lasso coefficients.  For both models, plot the t statistics of every connection using imagesc().  Ignoring that these subject samples are small for a proper analysis, which model seems to allow for better inferences when testing group differences?  What quality of the coefficients would make that model advantageous for testing group differences?
