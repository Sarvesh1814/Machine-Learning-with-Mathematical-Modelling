clc;
clear all;
close all;
%% 1) Generating 100 real numbers for variable X_test
s= rng
X_train = rand(100,1);
rng(s)
epsilon = (0.25).*rand(1,1);
Y_train = sin(2*pi*X_train)+ epsilon;

%% 2) Generating 50 real numbers for variable X_test

X_test = rand(50,1);
Y_test = sin(2*pi*X_test) + epsilon;

%% 3) Least Squares Polynomial Regression of order M=9 

A1 = Poly_AugMat(X_train,9);
lambda = regularized_fit(9,A1,Y_train);

u = Weights(lambda,A1,Y_train,9);
Y_pred = prediction(A1,u);

RMSE_P = RMSE(Y_train,Y_pred);
%training
figure;
scatter(X_train,Y_train,"Red+")
hold on;
scatter(X_train,Y_pred,"Black*")
legend("Actual","Prediction on Training set")
title("Using Least-Squares Polynomial Regression")

At1 = Poly_AugMat(X_test,9);
Y_predt = prediction(At1,u);
RMSE_Pt = RMSE(Y_test,Y_predt);
MAE_Pt = MAE(Y_test,Y_predt);
NMSE_Pt = NMSE(Y_test,Y_predt);
R2_Pt = R2(Y_test,Y_predt);

%testing
figure;
scatter(X_test,Y_test,"Red+")
hold on;
scatter(X_test,Y_predt,"Black*")
legend("Actual Test","Prediction on Test set")
title("Using Least-Squares Polynomial Regression")

%% 4) Gradient Descent 
iter=0;
theta = rand(10,1);
alpha = 0.002;

while true
    iter=iter+1;
    grad = ((lambda*theta)-(A1'*(Y_train-A1*theta)));
    theta = theta - alpha*grad;
    if norm(grad) <= 0.00001
        break
    end

end

Y_pred1 = A1*theta;
RMSE_GD = RMSE(Y_train,Y_pred1);

figure;
scatter(X_train,Y_train,"Red+")
hold on;
scatter(X_train,Y_pred1,"Black*")
legend("Actual","Predict")
title("Using Gradient Descent")

Y_predt1= At1*theta;
RMSE_GD_t = RMSE(Y_test,Y_predt1);
MAE_GD_t = MAE(Y_test,Y_predt1);
NMSE_GD_t = NMSE(Y_test,Y_predt1);
R2_GD_t = R2(Y_test,Y_predt1);

%testing
figure;
scatter(X_test,Y_test,"Red+")
hold on;
scatter(X_test,Y_predt1,"Black*")
legend("Actual Test","Prediction on Test set")
title("Using Gradient Descent")
%% 5)  Stochastic Gradient Descent

iter1=0;
theta1 = rand(10,1);
alpha1 = 0.02;
gradient = [];
while true
    Xs_train=[];
    Ys_train=[];
    r = randi([1,100],5,1);
    for i = 1:length(r)
        Xs_train(i,:) = X_train(r(i),:);
        Ys_train(i,:) = Y_train(r(i),:);
    end
    iter1=iter1+1;
    As1 = Poly_AugMat(Xs_train,9);
    grad1 = ((lambda*theta1)-(As1'*(Ys_train-As1*theta1)));
    theta1 = theta1 - alpha1*grad1;
    alpha1 = alpha1/1.0000005;
    gradient(iter1)= norm(grad1,2);
    if gradient(iter1) <= 0.01
        break
    end

end

Y_pred2 = A1*theta1;
RMSE_SGD = RMSE(Y_train,Y_pred2);

%training
figure;
scatter(X_train,Y_train,"Red+")
hold on;
scatter(X_train,Y_pred2,"Black*")
legend("Actual","Predict")
title("Using Stocastic Gradient Descent")

Y_predt2= At1*theta1;
RMSE_SGD_t = RMSE(Y_test,Y_predt2);
MAE_sGD_t = MAE(Y_test,Y_predt2);
NMSE_sGD_t = NMSE(Y_test,Y_predt2);
R2_sGD_t = R2(Y_test,Y_predt2);

%testing
figure;
scatter(X_test,Y_test,"Red+")
hold on;
scatter(X_test,Y_predt1,"Black*")
legend("Actual Test","Prediction on Test set")
title("Using Stochastic Gradient Descent")

figure;
plot(gradient,"Red");
title("Norm of Gradient at each epoch in Stochastic Gradient Descent")

%% 6) Least Squares Kernel Regression 
[lambda,sigma,lambda_sigma_grid] = LAMBDA(X_train,Y_train);
h1 = K_Mat(X_train,X_train,sigma);
V1 = Kernel_weights(lambda,h1,Y_train);
y_pred3 = h1*V1;
RMSE_LSK = RMSE(Y_train,y_pred3);

%training
figure;
scatter(X_train,Y_train,"Red+")
hold on;
scatter(X_train,y_pred3,"Black*")
legend("Actual","Predict")
title("Using Least Square Kernel Regression")

ht1 = K_Mat(X_test,X_train,sigma);
Y_predt3= ht1*V1;
RMSE_LSK_t = RMSE(Y_test,Y_predt3);
MAE_LSK_t = MAE(Y_test,Y_predt3);
NMSE_LSK_t = NMSE(Y_test,Y_predt3);
R2_LSK_t = R2(Y_test,Y_predt3);

%testing
figure;
scatter(X_test,Y_test,"Red+")
hold on;
scatter(X_test,Y_predt3,"Black*")
legend("Actual Test","Prediction on Test set")
title("Using Least Square Kernel Regression")

%% 7)Kernel Regression with stochastic gradient descent method.
iter2= 0;
alpha2 = 0.02;
v2= rand(101,1);
while true
    Xs_train=[];
    Ys_train=[];
    r = randi([1,100],5,1);
    for i = 1:length(r)
        Xs_train(i,:) = X_train(r(i),:);
        Ys_train(i,:) = Y_train(r(i),:);
    end
    iter2=iter2+1;
    h2 = K_Mat(Xs_train,X_train,sigma);
    grad2 = ((lambda*v2)-(h2'*(Ys_train-h2*v2)));
    v2= v2 - alpha2*grad2;
    alpha2 = alpha2/1.0000005;    
    if norm(grad2,2)<= 0.01
        break
    end

end

h22 = K_Mat(X_train,X_train,sigma);
Y_pred4 = h22*v2;
RMSE_KSGD = RMSE(Y_train,Y_pred4);

%training
figure;
scatter(X_train,Y_train,"Red+")
hold on;
scatter(X_train,Y_pred4,"Black*")
legend("Actual","Predict")
title("Using Kernel Stocastic Gradient Descent")

h23 = K_Mat(X_test,X_train,sigma);
Y_predt4= h23*v2;
RMSE_KSGD_t = RMSE(Y_test,Y_predt4);
MAE_KSGD_t = MAE(Y_test,Y_predt4);
NMSE_KSGD_t = NMSE(Y_test,Y_predt4);
R2_KSGD_t = R2(Y_test,Y_predt4);

%testing
figure;
scatter(X_test,Y_test,"Red+")
hold on;
scatter(X_test,Y_predt1,"Black*")
legend("Actual Test","Prediction on Test set")
title("Using Kernel Stochastic Gradient Descent")


%% 8) Least Square Polynomial Regression with modified data.
rnd = randi([1,100],5,1);
new_X=[];
new_Y=[];
for i = 1:length(X_train)
    if ismember(i,rnd)
        new_X(i,:)= X_train(i,:);
        temp= Y_train(i,:);
        new_Y(i,:)= temp*20;
    else
        new_X(i,:)= X_train(i,:);
        new_Y(i,:)= Y_train(i,:);
    end
end

A8 = Poly_AugMat(new_X,9);
lambda8= regularized_fit(9,A8,new_Y);
u8=Weights(lambda8,A8,new_Y,9);
Y_pred8 = prediction(A8,u8);

RMSE_8_t = RMSE(Y_train,Y_pred8);
MAE_8_t = MAE(Y_train,Y_pred8);
NMSE_8_t = NMSE(Y_train,Y_pred8);
R2_8_t = R2(Y_train,Y_pred8);

figure;
scatter(new_X,new_Y,"Red+");
hold on;
scatter(new_X,Y_pred8,"black*");
legend("Actual Training Set","Prediction on Training set")
title("Using Least Square Polynomial Regression with scaling values")

%% User defined functions

function  A = Poly_AugMat(X,M)
    for index = 1:10
        if index<= M
            A(:,index) = X.^index;
        else
            A(:,index) = 1;
        end
    end
end

% Finding best regularized fit
function lambda= regularized_fit(M,A,y)
    lambda_array = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7];
    min=1;
    RMSE_array = [];
    for index1 = 1:length(lambda_array)
        lambda = lambda_array(index1);
        u = Weights(lambda,A,y,M);
        y_pred = prediction(A,u);
        RMSE_P = RMSE(y,y_pred);
        RMSE_array(index1) = RMSE_P;
        if RMSE_P < RMSE_array(min)
            min= index1;
        end
    end
    lambda = lambda_array(min);
end

% Weight Finding Function
function u = Weights(lambda,A,Y,M)
    identity = eye(M+1);
    u = inv(lambda*identity+A'*A)*A'*Y;
end

% Predicting Function
function Y_pred= prediction(A,u)
    Y_pred = A*u;
end

% Root Mean Squre Function
function rmse = RMSE(Y,Y_pred)
    rmse = (sum((Y-Y_pred).^2)/length(Y)).^(0.5);
end

%NMSE
function nmse = NMSE(Y,Y_pred)
    Y_mean = sum(Y)/length(Y);
    nmse= sum(((Y-Y_pred).^2))/sum(((Y-Y_mean).^2));
end

%MAE
function mae= MAE(Y,Y_pred)
    mae = sum(Y-Y_pred)/length(Y);
end

%R2
function r2 = R2(Y,Y_pred)
    Y_mean = sum(Y)/length(Y);
    Y_pred_mean = sum(Y_pred)/length(Y_pred);
    r2= sum(((Y_pred-Y_pred_mean).^2))/sum(((Y-Y_mean).^2));
end

% lambda sigma grid search
function [lambda,sigma,lambda_sigma_grid] = LAMBDA(X,Y)
    lambda_array = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7];
    sigma_array=[2^-12,2^-11,2^-10,2^-9,2^-8,2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7,2^8,2^9,2^10,2^11,2^12];
    rt= 9999999999999999999;
    for lam = 1:length(lambda_array)
        L = lambda_array(lam);
        for sig = 1:length(sigma_array)
            S = sigma_array(sig);
            h = K_Mat(X,X,S);
            v = inv(L*eye(length(h))+h'*h)*h'*Y;
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
            h(i,j)= exp(-(norm_value/(sigma.^2))); %RBF Kernel
        end
        h(i,(length(d_j)+1))= 1;
    end
end

% Kernal Weights

function V = Kernel_weights(lambda,H,y)
    V = inv(lambda.*eye(101)+H'*H)*H'*y;
end