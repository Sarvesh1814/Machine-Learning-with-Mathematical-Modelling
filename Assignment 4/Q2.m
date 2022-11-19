%% Clearing all the past outputs and commands
clc;
clear all;
close all;

%% Motorcycle Training Data

motorcycle = readmatrix("motorcycle.csv");
X2= motorcycle(:,1);
Y2= motorcycle(:,2);

%% Support Vector Regression 


EPS2 = 0.1;
[lambda4,sigma4] = regularized_svm_fit(X2,Y2,EPS2);
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
    max_iter4 = 40000;  % max no of iterations
    alpha4 = 0.1;   % Step Size 
    
    while iter4 <= max_iter4 % stopping condition
        grad = (lambda4*V4)- (H4'*sign1(Y_train3-(H4*V4),EPS2));
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

figure;
plot(X_train3,Y_train3,"red")
hold on;
plot(X_train3,X2_pred_train2,"black")
plot(X_train3,X2_pred_train2+EPS2,"Green")
plot(X_train3,X2_pred_train2-EPS2,"Green")
legend("Actual","Predicted","+ε limit","-ε limit")
title("Support Vector Regression using leave one out method (Motorcycle Dataset")


%% user defined functions

% Hyperparameter Tuning for SVM
function [lambda,sig] = regularized_svm_fit(x,y,epsilon)
    lambda_array = [2^-10,2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7];
    sigma = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7];
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
            end
        end    

    end
    lambda = lambda_array(min);
    sig= sigma(min1);
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