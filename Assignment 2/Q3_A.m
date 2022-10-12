%% Reading the data and Initialization

%Reading Dataset
motorcycle = readmatrix("motorcycle.csv");
X= motorcycle(:,1);
Y= motorcycle(:,2);


%regularization parameter values


% Initializing a and b for sigmoid basis function
a=rand(10,1);
b=rand(10,1);
a=a';
b=b';

%% Train Test Splitting
[X_train,Y_train,X_test,Y_test]=train_test_split(X,Y,25);


%% Model 1 n =2

A2=AugMaT(X_train,2,a,b);
lambda2= regularized_fit(2,A2,Y_train);
u1 = Weights(lambda2,A2,Y_train,2);
Y_pred = prediction(A2,u1);

rmse2= RMSE(Y_train,Y_pred);
At2= AugMaT(X_test,2,a,b);
Yt_Pred = prediction(At2,u1);
MAE1 = sum(Y_train-Y_pred)/length(Y_train);


% plotting the model performance on data points
figure;
scatter(X_train,Y_train,'red+');
hold on;
scatter(X_train,Y_pred,'black+');
%figure;
hold on;
scatter(X_test,Y_test,'blue*');
hold on;
scatter(X_test,Yt_Pred,'m*');
legend("X/Y-actual","X/Y-Prediction","X_t/Y_t-test","X_t/Y_t-predicition")
title("Model for n=2")
xlabel("X")
ylabel("Y")
%% Model 2 n=5

A5=AugMaT(X_train,5,a,b);
lambda5= regularized_fit(5,A5,Y_train);
u5 = Weights(lambda5,A5,Y_train,5);
Y_pred5 = prediction(A5,u5);

rmse5= RMSE(Y_train,Y_pred5);
At5= AugMaT(X_test,5,a,b);
Yt_Pred5 = prediction(At5,u5);
MAE5 = sum(Y_train-Y_pred5)/length(Y_train);


% plotting the model performance on data points
figure;
scatter(X_train,Y_train,'red+');
hold on;
scatter(X_train,Y_pred5,'black+');
%figure;
hold on;
scatter(X_test,Y_test,'blue*');
hold on;
scatter(X_test,Yt_Pred5,'m*');
legend("X/Y-actual","X/Y-Prediction","X_t/Y_t-test","X_t/Y_t-predicition")
title("Model for n=5")
xlabel("X")
ylabel("Y")

%% Model3 n =10
A10=AugMaT(X_train,10,a,b);
lambda10= regularized_fit(10,A10,Y_train);

A10=AugMaT(X_train,10,a,b);
lambda10= regularized_fit(10,A10,Y_train);
u10 = Weights(lambda10,A10,Y_train,10);

Y_pred10 = prediction(A10,u10);
rmse10= RMSE(Y_train,Y_pred10);
MAE10 = sum(Y_train-Y_pred10)/length(Y_train);

At10= AugMaT(X_test,10,a,b);
Yt_Pred10= prediction(At10,u10);

% plotting the model performance on data points

figure;
scatter(X_train,Y_train,'red+');
hold on;
scatter(X_train,Y_pred10,'black+');
%figure;
hold on;
scatter(X_test,Y_test,'blue*');
hold on;
scatter(X_test,Yt_Pred10,'m*');
legend("X/Y-actual","X/Y-Prediction","X_t/Y_t-test","X_t/Y_t-predicition")
title("Model for n=10")
xlabel("X")
ylabel("Y")

%% Writing functions for repeatative codes 

%basis function
function ph = phi(x,a,b,n)   
    ph = 1/(1+exp(-(a(n)'*x+b(n))));
end

%Augmented Matrix 
function A = AugMaT(X,n,a,b)
    for i = 1:length(X)
        x=X(i,:);
        for j= 1:n+1
            if j == n+1
                A(i,j)=1;
            else
                A(i,j)=phi(x,a,b,j);
            end
        end
    end    

end

%train-test splitting 
function [X_train,Y_train,X_test,Y_test] = train_test_split(X,Y,n)
    xmin=1;
    xmax=length(X);
    x=xmin+rand(1,n)*(xmax-xmin);
    x=int16(x);
    a=1;
    b=1;

    for i= 1:length(X)
        if ismember(i,x)
            X_test(a,:)=X(i,:);
            Y_test(a,:)=Y(i,:);
            a=a+1;
        else
            X_train(b,:)=X(i,:);
            Y_train(b,:)=Y(i,:);
            b=b+1;
        end
    end
end

% Predicting Function
function Y_pred= prediction(A,u)
    Y_pred = A*u
end

% Weight Finding Function
function u = Weights(lambda,A,Y,n)
    identity = eye(n+1);
    u = inv(lambda*identity+A'*A)*A'*Y;
end

% Root Mean Squre Function
function rmse = RMSE(Y,Y_pred)
    rmse = (sum((Y-Y_pred).^2)/length(Y)).^(0.5);
end

% Finding best regularized fit
function lambda= regularized_fit(n,A,y)
    lambda_array = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7];
    min=1;
    RMSE_array = [];
    for index1 = 1:length(lambda_array)
        lambda = lambda_array(index1);
        u = Weights(lambda,A,y,n);
        y_pred = prediction(A,u);
        RMSE_P = RMSE(y,y_pred);
        RMSE_array(index1) = RMSE_P;
        if RMSE_P < RMSE_array(min)
            min= index1;
        end
    end
    lambda = lambda_array(min);
end

