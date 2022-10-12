%% Reading Data 
housing = readmatrix("bostonhousing.csv");
X= housing(1:500,1:13);
Y= housing(1:500,14);

%% Normalizing Data

% Normalizing the data will benifit the model training.
% for the 5th column the data is binary so there is no need of normalizing it 
for i = 1:13
    if i== 5 
        X_N(:,i)= X(:,i);
    else
        x=X(:,i);
        N_mean = sum(x)/length(x);
        N_sigma = (sum(((x-N_mean).^2))/length(x)).^0.5;
        X_N(:,i) = (x-N_mean)/N_sigma;
    end
end

%% Initializing the variables
lambda_array=[2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7];
sigma_array=[2^-12,2^-11,2^-10,2^-9,2^-8,2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7,2^8,2^9,2^10,2^11,2^12];
RMSE=[];
NMSE=[];
R2=[];
MAE=[];

%% finding best pair of lambda and Sigma to using grid search to get optimal performance
[lambda,sigma,lambda_sigma_grid] = LAMBDA(lambda_array,sigma_array,X_N,Y);


%% Ten-fold Cross-validation Algorithm 

tic % For Evaluating Training Time of the model 

% Splitting data into folds (10 folds) 

counter = 1;
for index = 1:10
    X_train =[];
    X_test = [];
    Y_train = [];
    Y_test = [];
    tst =1;
    trn=1;
    for index1 =1:length(X)
        if index1 == counter && tst <=50
            X_test(tst,:)= X(index1,:);
            Y_test(tst,:)= Y(index1,:);
            tst =tst+1;
            counter=counter+1;
        else
            if trn<=450
                X_train(trn,:)= X(index1,:);
                Y_train(trn,:)= Y(index1,:);
                trn = trn+1;
            end
        end
    end 
    
    % Processess for each fold 

    H = K_Mat(X_train,X_train,sigma);
    V = inv(lambda.*eye(451)+H'*H)*H'*Y_train; %Obtaining Weights  
    
    H_t = K_Mat(X_test,X_train,sigma);
    Y_pred = H_t*V; %prediction on test set (Size =50)

    % Evaluation metrics for each fold 

    %Root Mean Square Error
    RMSE(index) = (sum((Y_test-Y_pred).^2)/length(Y_test)).^(0.5);
    
    % Mean Absolute Error
    MAE(index) = sum(Y_test-Y_pred)/length(Y_test);
    
    %Normalized Mean Square Error
    Y_test_mean = sum(Y_test)/length(Y_test);
    NMSE(index) = sum(((Y_test-Y_pred).^2))/sum(((Y_test-Y_test_mean).^2));

    % R_square Error
    Y_pred_mean = sum(Y_pred)/length(Y_pred);
    R2(index) = sum(((Y_pred-Y_pred_mean).^2))/sum(((Y_test-Y_test_mean).^2));
    
end
toc   % end of time counter for training

%% Mean and Standard Deviation of all metrics
Mean_RMSE= sum(RMSE)/length(RMSE);
Std_RMSE = ((sum((RMSE - Mean_RMSE).^2))/length(RMSE)).^0.5;

Mean_MAE = sum(MAE)/length(MAE);
Std_MAE  = ((sum((MAE - Mean_MAE).^2))/length(MAE)).^0.5;

Mean_NMSE= sum(NMSE)/length(NMSE);
Std_NMSE = ((sum((NMSE - Mean_NMSE).^2))/length(NMSE)).^0.5;

Mean_R2= sum(R2)/length(R2);
Std_R2 = ((sum((R2 - Mean_R2).^2))/length(R2)).^0.5;
%% User Defined functions used in Algorithm

% Lambda_Sigma_GridSearch
function [lambda,sigma,lambda_sigma_grid] = LAMBDA(lambda_array,sigma_array,X,Y)
    
    rt= 9999999999999999999;
    for lam = 1:length(lambda_array)
        L = lambda_array(lam);
        for sig = 1:length(sigma_array)
            S = sigma_array(sig);
            h = K_Mat(X,X,S);
            v = inv(L*eye(501)+h'*h)*h'*Y;
            y_p= h*v;
            rmse = (sum((Y-y_p).^2)/length(Y)).^(0.5);
            lambda_sigma_grid(lam,sig) = rmse;
            if rmse < rt
                rt =rmse;
                lambda=lambda_array(lam);
                sigma = sigma_array(sig);
            end
        end
    end
    
end

% Kernel Matrix
function h= K_Mat(d_i,d_j,sigma)

    for i = 1:length(d_i)
        for j = 1:length(d_j)
            norm_vector = abs(d_i(i,:) - d_j(j,:));
            norm_value = ((sum(norm_vector.^2)).^0.5);
            h(i,j)= exp(-(norm_value/(2*(sigma.^2)))); %RBF Kernel
        end
        h(i,(length(d_j)+1))= 1;
    end
end




