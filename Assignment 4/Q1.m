%% Clearing all the past outputs and commands
clc;
clear all;
close all;

%% Random Training Data 

X_train1 = sort(rand(1000,1));
epsilon = 0.25.*(rand(1000,1)-0.5);
Y_train1 = sin(2*pi*X_train1)+ epsilon;

X_test1= sort(rand(50,1));
epsilon1 = (0.25).*(rand(50,1)-0.5);
Y_test1= sin(2*pi*X_test1)+ epsilon1;

%% Motorcycle Training Data

motorcycle = readmatrix("motorcycle.csv");
X2= motorcycle(:,1);
Y2= motorcycle(:,2);

%% Q1-A) L1 Norm Loss Kernel Regression

% Hyperparameter Tuning (Lambda and Sigma)
[lambda,sigma] = regularized_fit(X_train1,Y_train1);

% Kernel and Initializing weights
H1 = K_Mat(X_train1,X_train1,sigma);
V1 = rand(length(X_train1)+1,1);
iter1 = 0;  % iteration counter
max_iter1 = 10000;  % max no of iterations
alpha1 = 0.1;   % Step Size 

while iter1 <= max_iter1  % stopping condition
    
    grad = (lambda*V1)- (H1'*sign(Y_train1-(H1*V1))); % gradient
    V1 = V1- alpha1*grad; % updating kernel weights
    iter1=iter1+1 
    alpha1 = alpha1/(1+alpha1*iter1); % Simulated Annealing
end

X1_pred_train = H1*V1;
H2_t = K_Mat(X_test1,X_train1,sigma);
Xtest1_pred = H2_t*V1;

% Error Metrics on Train and Test
Train1_rmse = RMSE(Y_train1,X1_pred_train);
Test1_rmse  = RMSE(Y_test1,Xtest1_pred);

Train1_MAE = MAE(Y_train1,X1_pred_train);
Test1_MAE =  MAE(Y_test1,Xtest1_pred);

% Plotting the estimate on train set and test set

figure;
scatter(X_train1,Y_train1,"red+")
hold on;
plot(X_train1,X1_pred_train,"black")
legend("Actual data with noise","Estimated Curve (Predicted)")
title("L1 norm Kernel Regression (Training")

figure;
scatter(X_test1,Y_test1,"Blue+")
hold on;
plot(X_test1,Xtest1_pred,"Green")
legend("Actual data with noise","Estimated Curve (Predicted)")
title("L1 norm Kernel Regression (Testing with 50 data points)")

%% Q1-B) Support Vector Regression  


[lambda2,sigma2,EPS] = regularized_svm_fit(X_train1,Y_train1);
H2 = K_Mat(X_train1,X_train1,0.0078);
V2 = rand(length(X_train1)+1,1);
iter2 = 0;
max_iter2 = 20000;
lambda2 =   0.03125;
alpha2 = 0.1;

while iter2 <= max_iter2
    
    grad = (lambda2*V2)- (H2'*sign1(Y_train1-(H2*V2),EPS));
    V2 = V2- alpha2*grad;
    iter2=iter2+1
    alpha2 = alpha2/(1+alpha2*iter2); 
end

X1_pred2_train = H2*V2;
H2_t = K_Mat(X_test1,X_train1,sigma2);
Xtest2_pred = H2_t*V2;

Train2_rmse = RMSE(Y_train1,X1_pred2_train);
Train2_MAE = MAE(Y_train1,X1_pred2_train);

Test2_MAE= MAE(Y_test1,Xtest2_pred);
Test2_rmse  = RMSE(Y_test1,Xtest2_pred);

figure;
scatter(X_train1,Y_train1,"red+")
hold on;
plot(X_train1,X1_pred2_train,"black")
plot(X_train1,X1_pred2_train+EPS,"Green")
plot(X_train1,X1_pred2_train-EPS,"Green")
legend("Actual","Predicted","+ε limit","-ε limit")
title("Support Vector Regression (Training)")

figure;
scatter(X_test1,Y_test1,"red+")
hold on;
plot(X_test1,Xtest2_pred,"black")
scatter(X_test1,Xtest2_pred+EPS,"Green*")
scatter(X_test1,Xtest2_pred-EPS,"Green*")
legend("Actual","Predicted","+ε limit","-ε limit")
title("Support Vector Regression (Training)")

%%
sp1= sparsity_mat(V2)
%% Q2 L1 norm kernel regression model with leave one out method

[lambda3,sigma3] = regularized_fit(X2,Y2);

% lambda3 = 2^-12;
% sigma3  = 0.0078;
V3 = rand(133,1);

train_rmse = 1000;
test_rmse = 1000;

train_mae = 1000;
test_mae = 1000;


for i = 1:133
    i
    X_train2 = X2(setdiff(1:length(X2),i));
    X_test2 = X2(i);
    Y_train2 = Y2(setdiff(1:length(Y2),i));
    Y_test2 = Y2(i);

    H3= K_Mat(X_train2,X_train2,sigma3);
    iter3 = 0;  % iteration counter
    max_iter3 = 10000;  % max no of iterations
    alpha3 = 0.1;   % Step Size 
    
    while iter3 <= max_iter3  % stopping condition
        
        grad = (lambda3*V3)- (H3'*sign(Y_train2-(H3*V3))); % gradient
        V3 = V3- alpha3*grad; % updating kernel weights
        iter3=iter3+1;
        alpha3 = alpha3/(1+alpha3*iter3); % Simulated Annealing
    end
    
    pred_train = H3*V3;
    Train2_rmse = RMSE(Y_train2,pred_train);
    Train2_MAE = MAE(Y_train2,pred_train);


    H3_t = K_Mat(X_test2,X_train2,sigma3);
    test_pred = H3_t*V3;
    Test2_MAE= MAE(Y_test2,test_pred);
    Test2_rmse  = RMSE(Y_test2,test_pred);
    
    if train_mae > Train2_MAE
        train_mae= Train2_MAE;
    end

    if train_rmse > Train2_rmse
        train_rmse= Train2_rmse;
    end

    if test_mae > Test2_MAE
        test_mae= Test2_MAE;
    end

    if test_rmse > Test2_rmse
        test_rmse= Test2_rmse;
    end


end

X2_pred_train = H3*V3;

figure;
scatter(X_train2,Y_train2,"red+")
hold on;
plot(X_train2,X2_pred_train,"black")
legend("Actual data with noise","Estimated Curve (Predicted)")
title("L1 norm Kernel Regression with Leave one out method (Motorcycle Dataset) ")


%% Q3 Epsilon - Support Vector Regression with leave one out method


[lambda4,sigma4,epsilon3] = regularized_svm_fit(X2,Y2);
V4= rand(133,1);

train2_rmse = 1000;
test2_rmse = 1000;

train2_mae = 1000;
test2_mae = 1000;


for i = 1:133
    i
    X_train3 = X2(setdiff(1:length(X2),i));
    X_test3 = X2(i);
    Y_train3 = Y2(setdiff(1:length(Y2),i));
    Y_test3 = Y2(i);

    H4= K_Mat(X_train3,X_train3,sigma4);
    iter4 = 0;  % iteration counter
    max_iter4 = 50000;  % max no of iterations
    alpha4 = 0.1;   % Step Size 
    
    while iter4 <= max_iter4 % stopping condition
        grad = (lambda4*V4)- (H4'*sign1(Y_train3-(H4*V4),epsilon3));
        V4 = V4- alpha4*grad;
        iter4=iter4+1;
        alpha4 = alpha4/(1+alpha4*iter4); 
    end

    pred_train2 = H4*V4;
    Train2_rmse2 = RMSE(Y_train3,pred_train2);
    Train2_MAE2 = MAE(Y_train3,pred_train2);


    H4_t = K_Mat(X_test3,X_train3,sigma4);
    test2_pred = H4_t*V4;
    Test2_MAE2= MAE(Y_test3,test2_pred);
    Test2_rmse2  = RMSE(Y_test3,test2_pred);
    
    if train2_mae > Train2_MAE2
        train2_mae= Train2_MAE2;
    end

    if train2_rmse > Train2_rmse2
        train2_rmse= Train2_rmse2;
    end

    if test2_mae > Test2_MAE2
        test2_mae= Test2_MAE2;
    end

    if test2_rmse > Test2_rmse2
        test2_rmse= Test2_rmse2;
    end


end

X2_pred_train2 = H4*V4;

sp = sparsity_mat(V4); % In percentage

figure;
scatter(X_train3,Y_train3,"red*")
hold on;
plot(X_train3,X2_pred_train2,"black")
plot(X_train3,X2_pred_train2-epsilon3,"Green")
plot(X_train3,X2_pred_train2+epsilon3,"Green")
legend("Actual","Predicted","+ε limit","-ε limit")
title("Support Vector Regression using leave one out method (Motorcycle Dataset)")

%% user defined functions
function  A = Poly_AugMat(X,M)
    for index = 1:M+1
        if index<= M
            A(:,index) = X.^index;
        else
            A(:,index) = 1;
        end
    end
end

% For L1 norm loss kernerl Regression
function [lambda,sig] = regularized_fit(x,y)
    lambda_array = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7];
    sigma = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7];
    min2=1
    min=1;
    min1=1;
    RMSE_array = [];
    RMSE_min=1000000000;
    
    for index = 1:length(lambda_array)
        lambda1 = lambda_array(index);
        
        for index1 = 1:length(sigma)
            H = K_Mat(x,x,sigma(index1));
            iters=0;
            alpha1 = 0.01;
            V=ones(length(x)+1,1);
            while iters <= 1000
                grad = (lambda1*V)- (H'*sign(y-(H*V)));
                V = V- alpha1*grad;
                iters=iters+1;
                alpha1 = alpha1/(1+alpha1*iters); %Simulated Annealing
            end

            y_pred = H*V;

            RMSE_P = RMSE(y,y_pred);
            RMSE_array(index1) = RMSE_P;

            if RMSE_P < RMSE_min
                RMSE_min= RMSE_P;
                min= index;
                min1=index1;
                
            end
        end    

    end
    lambda = lambda_array(min);
    sig= sigma(min1);
    

end

% Hyperparameter Tuning for SVM
function [lambda,sig,epsilon] = regularized_svm_fit(x,y)
    epsilon_array= [0.001,0.01,0.1,1,10];
    lambda_array = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7];
    sigma = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7];
    min=1;
    min1=1;
    min2=1;
    RMSE_array = [];
    RMSE_min=1000000000;
    
    for index2 = 1:length(epsilon_array)
        epsilon = epsilon_array(index2);
        for index = 1:length(lambda_array)
            lambda1 = lambda_array(index);
            
            for index1 = 1:length(sigma)
                H = K_Mat(x,x,sigma(index1));
                iters=0;
                alpha1 = 0.01;
                V=ones(length(x)+1,1);
                while iters <= 1000
                    grad = (lambda1*V)- (H'*sign1(y-(H*V),epsilon));
                    V = V- alpha1*grad;
                    iters=iters+1;
                    alpha1 = alpha1/(1+alpha1*iters); %Simulated Annealing
                end
    
                y_pred = H*V;
    
                RMSE_P = RMSE(y,y_pred);
                RMSE_array(index1) = RMSE_P;
    
                if RMSE_P < RMSE_min
                    RMSE_min= RMSE_P;
                    min= index;
                    min1=index1;
                    min2=index2;
                end
            end    
    
        end
        lambda = lambda_array(min);
        sig= sigma(min1);
        epsilon= epsilon_array(min2);
    end
end


% Root Mean Squre Function
function rmse = RMSE(Y,Y_pred)
    rmse = (sum((Y-Y_pred).^2)/length(Y)).^(0.5);
end


%MAE
function mae= MAE(Y,Y_pred)
    mae = sum(Y-Y_pred)/length(Y);
end

% Kernel Matrix
function h= K_Mat(X2,X1,sigma)
    h = ones(length(X2),length(X1)+1);
    for m = 1:length(X2)
        for n = 1:length(X1)
            h(m,n) = exp(-(X1(n)-X2(m))'*(X1(n)-X2(m))/sigma);
        end
    end
end

% Epsilion Signum Function for epsilion-SVR
function sig = sign1(X,epsilon)
    sig = ones(length(X),1);
    for m = 1:length(X)
        if X(m)>epsilon
            sig(m) =1;
        elseif X(m)< -epsilon
            sig(m)=-1;
        else
            sig(m)=0;
        end
    end
end

function sp = sparsity_mat(V)
    s = 0;
    for a = 1:length(V)
        if V(a) <0.01 && V(a)>-0.01
            s=s+1;
        end
    end
    sp = (s/length(V))*100;
end



