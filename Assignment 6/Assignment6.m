clc;
clear all;
close all;

%% Logistic Regression

mu_A = [-1 1];
Sigma_A = [0.7 0; 0.0 0.3];
X_A = mvnrnd(mu_A,Sigma_A,1750);
t1 = mvnrnd(mu_A,Sigma_A,250);
Y_A = zeros(1750,1);
Y_1 = zeros(250,1);


mu_B = [2 2];
Sigma_B = [0.7 0; 0.0 0.3];
X_B = mvnrnd(mu_B,Sigma_B,1750);
t2 = mvnrnd(mu_B,Sigma_B,250);
Y_B = ones(1750,1);
Y_2 = ones(250,1);

% Initialization of Weights
n = size(X_A);
n = n(2)+1;
w = zeros(n,1);

%Iterations and Alpha Value
iter = 0;
alpha = 0.00001;


%training set
X= [X_A;X_B];
X= [X,ones(3500,1)];
Y= [Y_A;Y_B];

%test set
X_test= [t1;t2];
X_test= [X_test,ones(500,1)];
Y_test= [Y_1;Y_2];
labels = [0,1];
lambda = LambdaTune(X,Y,alpha,labels);
disp("Tuning Complete")
[w,iter] = LogisticRegression(X,Y,w,lambda,alpha,50000,0.001);
disp("Training Complete")
% prediction on the test set

f_x = Sigmoid(w,X_test);
label = Classifier(f_x,0.5,labels);
[X1,X2] = discriminant_Logistic(w);
[X3,X4] = discriminant_Bayesian(X_A,X_B,mu_A,mu_B,Sigma_A,Sigma_B);
accuracy = Acc(Y_test,label);
[precision,recall,f1_score,confusion_matrix] = Meterics(Y_test,label);

disp("Total number of iteration in Gradient Descent:"+ iter)
disp("Accuracy of Logistic Regression on Test Set: "+accuracy+"%" )

disp(" ")
disp("Precision of given model:"+ precision)
disp("Recall of given model: "+ recall)
disp("F1-Score of given model: "+ f1_score)
disp("Confusion of given model: ")
confusion_matrix


%plotting the output

figure;
scatter(X_A(:,1),X_A(:,2),10,"red","filled")
hold on;
scatter(X_B(:,1),X_B(:,2),10,"blue","filled")
hold on;
scatter(X1,X2,"black","filled")
hold on;
scatter(X3,X4,"Green","filled")
xlim([-5 5]) 
ylim([-5 5])
legend("Class: 0","Class:1","Logistic Regression","Bayesian Classifier")
title("Logistic Regression")

%% Logistic Regression on IRIS Dataset
iris = readmatrix("IRIS.csv");
X=iris(1:100,2:5);
Y=iris(1:100,1);
X=[X,ones(length(X),1)];
%train-validation-test split
[X_train,Y_train,X_check,Y_check] = train_test_split(X,Y,20);
[X_val,Y_val,X_test,Y_test] = train_test_split(X_check,Y_check,50);

alpha = 0.00000007;
labels=[1,2];
% Initialization of Weights
n = size(X);
n = n(2);
w = zeros(n,1);

%hyperparameter tuning
lambda = LambdaTune(X_val,Y_val,alpha,labels);
disp("Tuning Complete")
[w,iter] = LogisticRegression(X_train,Y_train,w,lambda,alpha,10000000,0.2);
disp("Training Complete")   
% prediction on the test set

f_x = Sigmoid(w,X_train);
label = Classifier(f_x,1.5,labels)';
accuracy_train = Acc(Y_train,label);
disp("Total number of iteration in Gradient Descent:"+ iter)
disp("Accuracy of Logistic Regression on Training Set: "+accuracy_train+"%" )

f_x1 = Sigmoid(w,X_test);
label1 = Classifier(f_x1,1.5,labels)';
accuracy_test = Acc(Y_test,label1);
disp("Accuracy of Logistic Regression on Test Set: "+accuracy_test+"%" )


%% Function
function lambda_final = LambdaTune(X,Y,alpha,labels)
    n = size(X);
    n = n(2);
    w = zeros(n,1);
    alpha = alpha;
    lambda_array = [2.^-12,2.^-11,2.^-10,2.^-9,2.^-8,2.^-7,2.^-6,2.^-5,2.^-4,2.^-3,2.^-2,2.^-1,2.^1,2.^2,2.^3,2.^4,2.^5,2.^6,2.^7,2.^8];
    accuracy_array=[];
    acc=0;
    lambda_final = 0;
    for i = 1:length(lambda_array)
        lambda = lambda_array(i);
        
        [w,iter]=LogisticRegression(X,Y,w,lambda,alpha,10000,0.01);
        f_x = Sigmoid(w,X);
        label = Classifier(f_x,0.5,labels);
        accuracy = Acc(Y,label);
        if accuracy> acc
            acc = accuracy;
            lambda_final = lambda;
        end
        accuracy_array(i)= accuracy;
        
    end
    accuracy_array;
    disp("Best Tuning Accuracy on  validation set: "+acc+"%"+" with the lambda value: "+lambda_final)
end

function [w,iter] = LogisticRegression(X,Y,w,lambda,alpha,max_iter,epsilion)
    iter = 0;
    while iter<max_iter       
        iter = iter+1;
        grad = ((X'*(Y-Sigmoid(w,X)))+ lambda*w);
        w = w +alpha*grad;
        Norm=norm(grad);
        if norm(grad) <= epsilion
           
            break 
        end
        
    end
    Norm;
end 

function X = Acc(test,actual)
    a=0;
    y=length(test);
    for i = 1:y
        if test(i)== actual(i)
            a=a+1;
        end
    end
    X = (a/y)*100;
end

function [precision,recall,f1_score,confusion_matrix] = Meterics(test,actual)
    a=0;
    TP = 0;
    FP=0;
    TN=0;
    FN=0;
    y=length(test);
    for i = 1:y
        if actual(i) == 1
            if test(i) ==1
                TP=TP+1;
            else
                FN = FN+1;
            end
        else
            if test(i)==0
                TN=TN+1;
            else
                FP = FP+1;
            end
        end
    end
    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    f1_score = 2*((precision*recall)/(precision+recall));
    confusion_matrix = [TP,FN;FP,TN];
end


function f_x = Sigmoid(W,X)
    
    f_x = exp(X*W)/ 1+ exp(X*W);
end

function pred_labels = Classifier(f_x,seperator_value,labels)
    pred_labels = [];

    for i = 1:length(f_x)
        if f_x(i)> seperator_value
            pred_labels(i) = labels(2);
        else
            pred_labels(i) = labels(1);
        end
    end
end

function [X1,X2] = discriminant_Logistic(w)

    X1 = linspace(-5,5,300)';
    X2 = -w(3)-(w(1)*X1)/w(2);
    
end

function [X1,X2] = discriminant_Bayesian(X_A,X_B,mu_A,mu_B,sigma_A,sigma_B)
    if sigma_A == sigma_B
        if (sigma_A(1) == sigma_A(4)) and (sigma_A(2) == sigma_A(4)) 
            d = ((mu_A'-mu_B')'*(mu_A'+mu_B'))/2;
             m = (mu_A'-mu_B')';
     
            if m(2) == 0
                X1 =zeros(100,1);
                X2 = linspace(min(min(X_A)),max(max(X_A)));
            else
                X1= linspace(min(X_A),max(X_A));
                X2 = ((d*ones(100,1)-(X1*m(1))))/m(2);
            end 
        
        else
            d = (inv(sigma_A)*(mu_A'-mu_B'))'*(mu_A'+mu_B')*0.5;
            m = (inv(sigma_A)*(mu_A'-mu_B'))';

            if m(2) == 0
                X1 = zeros(300,1);
                X2 = linspace(min(min(X_A)),max(max(X_A)));
            else
                X1= linspace(-5,5,300);
                X2 = ((d*ones(300,1)-(X1*m(1))'))/m(2);
            end
        end
    else
        A1 = -0.5*inv(sigma_A)+0.5*inv(sigma_B);
        A2 = (inv(sigma_A)*mu_A' - inv(sigma_B)*mu_B')' ;
        A3= -0.5*mu_A*inv(sigma_A)*mu_A' + 0.5*mu_B*inv(sigma_B)*mu_B'-0.5*log(norm(sigma_A))+0.5*log(norm(sigma_B));
        
        xx1 = linspace(min(min(X_A)),max(max(X_A)),200)';
        l=size(xx1);
        X5=[];
        X6=[];
        index = 0;
    
        for i = 1: l(1)
            x1 = xx1(i);
            A = A1(2,2);
            B = x1*(A1(1,2)+A1(2,1)) + A2(1,2);
            C = A3+ (A2(1,1)*x1);
        
            temp = (-B + (B.^2 - 4*A*C).^0.5)/(2*A);
            
            if isreal(temp)
                index = index +1;
                X5(index) = x1;
                X6(index) = temp;
            end
        
            temp = (-B - (B.^2 - 4*A*C).^0.5)/(2*A);
            
            if isreal(temp)
                index = index +1;
                X5(index) = x1;
                X6(index) = temp;
            end
        end
    end
end


function [X_train,Y_train,X_test,Y_test] = train_test_split(X,Y,n)
    n=n/100;
    l=[];
    lim=n*length(X);
    lim = ceil(lim);
    j=1;
    a=1;
    b=1;
    while j<= lim
        t = rand(1,1)*length(X);
        t = int16(t);
        if ismember(t,l)
        
        else
            l(j) = t;
            j=j+1;
        end
    end
    for i = 1:length(X)
        
        if ismember(i,l)
            X_test(a,:)=X(i,:);
            Y_test(a,:)=Y(i,1);
            a=a+1;
        else
            X_train(b,:)=X(i,:);
            Y_train(b,:)=Y(i,1);
            b=b+1;
        end
    end
end
