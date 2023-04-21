function [B,FitInfo] = ridgeCV(X,y,NameValueArgs)
% RIDGECV Performs ridge regression with cross validation to find the best
% hyperparameter value.
%   Uses the original 'ridge' function to perform regression. Quantifies
%   model performance using MSE. Inputs and outputs are meant to resemble
%   those for the 'lasso' function. 
%   *CV - number of fold to use for k-fold cross validation (default: 10)
%   *Lambda - scalar or vector of hyperparameter values to use (default:
%   10^(-5:5))
%   *nrSweeps - number of sweeps to perform in narrowing range of optimal
%   lambdas (default: 2, yields a power of 10 with the exponent precise to
%   the first decimal)

arguments
    X                       (:,:) double
    y                       (:,1) double {mustBeEqualSizeDim1(X,y)}
    NameValueArgs.CV        (1,1) {mustBeInteger,mustBePositive} = 10
    NameValueArgs.Lambda    (:,:) double {mustBeNonnegative} = []
    NameValueArgs.nrSweeps  (1,1) {mustBeInteger,mustBePositive} = 2
end

NR_ITERS_MAX = 100; % Maximum number of iterations to run before quitting loop
LAMBDAS_POWER10_RANGE = 5; % Number of lambdas above and below current best to test on each sweep
ZERO_COEFS_TOL = 10^(-3); % Threshold for considering a coefficient nonzero

nrObs = size(X,1); % Observations
nrVars = size(X,2); % Predictor variables

% Create a boolean array to divide observations between CV folds
nrFolds = NameValueArgs.CV;
kFoldsIdx = false(nrObs,nrFolds);
k = 1;
for i = 1:nrObs
    kFoldsIdx(i,k) = true;
    if k == nrFolds
        k = 1;
    else
        k = k+1;
    end
end

% Function to calculate MSE from actual and predicted data
calcMSE = @(yActual,yPredicted) mean((repmat(yActual,1,size(yPredicted,2)) - yPredicted).^2,1);

% If user provided lambda values...
if ~isempty(NameValueArgs.Lambda)
    lambdas = NameValueArgs.Lambda;
    MSEs = nan(length(NameValueArgs.Lambda),nrFolds);
    
    % Loop over folds to fit models and calculate MSE for all lambdas
    for k = 1:nrFolds
        betasCV = ridge(y(~kFoldsIdx(:,k)),X(~kFoldsIdx(:,k),:),lambdas);
        yPred = X(kFoldsIdx(:,k),:)*betasCV;
        MSEs(:,k) = mean((repmat(y(kFoldsIdx(:,k)),1,length(lambdas)) - yPred).^2,1);
    end
    MSEs = mean(MSEs,2);

% If user did not provide lambda values...
else
    % Since the final number of lambdas is unknown, variables will be
    % allowed to expand rather than being preallocated
    lambdas = [];
    MSEs = nan(0,nrFolds);
    
    % Initialize loop to search and test lambdas
    sweep = 0; % How many times the lambda values have been made more precise
    iter = 0; % Total number of iterations - will quit loop if limit reached
    extendSweep = 0; % If next loop has to search lambdas larger or smaller than in previous range
    bestPower10 = 0; % Current optimal lambda

    while sweep < NameValueArgs.nrSweeps % Loop until desired precision is reached
        if extendSweep == 0 % Best lambda was in range- add precision
            lambdasPower10 = bestPower10 + (-LAMBDAS_POWER10_RANGE:LAMBDAS_POWER10_RANGE)'/(10^sweep);
        elseif extendSweep == -1 % Best lambda was at lower edge of range- search lower values
            lambdasPower10 = bestPower10 + (-LAMBDAS_POWER10_RANGE:1)'/(10^sweep);
        elseif extendSweep == 1 % Best lambda was at higher edge of range- search higher values
            lambdasPower10 = bestPower10 + (-1:LAMBDAS_POWER10_RANGE)'/(10^sweep);
        end
        
        % Lambdas to test in this loop
        newLambdas = 10.^lambdasPower10;

        % Loop over folds to fit models and calculate MSE for new lambdas
        newMSEs = nan(length(newLambdas),nrFolds);
        betasCV = nan(nrVars,length(newLambdas),nrFolds);
        for k = 1:nrFolds
            betasCV(:,:,k) = ridge(y(~kFoldsIdx(:,k)),X(~kFoldsIdx(:,k),:),newLambdas);
            yPred = X(kFoldsIdx(:,k),:)*betasCV(:,:,k);
            newMSEs(:,k) = calcMSE(y(kFoldsIdx(:,k)),yPred);
        end
        newMSEs = mean(newMSEs,2);
        minMSEIdx = find(newMSEs==min(newMSEs),1);
        
        % Append newest lambdas and MSEs to full sets
        MSEs = [MSEs;newMSEs];
        lambdas = [lambdas;newLambdas];

        % Check that the newest lambda values are yielding nonzero coefficients
        hasNonzeroCoefs = any(abs(mean(betasCV,3)) > ZERO_COEFS_TOL,1);
        if hasNonzeroCoefs(minMSEIdx) % If best lambda gives nonzeros coeffs...
            if minMSEIdx==1 % If best lambda is lowest in range, test even lower values
                extendSweep = -1;
            elseif minMSEIdx==length(newMSEs) % If best lambda is highest in range, test even higher values
                extendSweep = 1;
            else % If best lambda is in middle of range, increase precision
                extendSweep = 0;
                sweep = sweep+1;
            end
            bestPower10 = lambdasPower10(minMSEIdx); % Exponent of current best lambda

        else % If the best lambda is producing all zeros coeffs, warn user and quit the loop
            warning('Lambda optimization converged on a model with only zero coefficients')
            break
        end

        iter = iter+1; % Track total number of iterations
        if iter>= NR_ITERS_MAX
            break
        end
    end
    
    % Put lambdas in ascending order
    [lambdas,sortIdx] = sort(lambdas);
    MSEs = MSEs(sortIdx);
end

% Assemble output structure
FitInfo.Lambda = reshape(lambdas,1,length(lambdas));
FitInfo.MSE = MSEs;
FitInfo.IndexMinMSE = find(MSEs==min(MSEs),1);
FitInfo.LambdaMinMSE = FitInfo.Lambda(FitInfo.IndexMinMSE);

% Calculate betas for all lambdas using entire dataset
B = ridge(y,X,FitInfo.Lambda);

end


function mustBeEqualSizeDim1(a,b)
    if size(a,1)~=size(b,1)
        eid = 'SizeDim1:notEqual';
        msg = 'Size of first input must equal size of second input along the first dimension.';
        throwAsCaller(MException(eid,msg))
    end
end