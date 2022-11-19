%% Logistic Regression on IRIS Dataset
iris = readmatrix("IRIS.csv");
X=iris(1:100,2:5);
Y=iris(1:100,1);
X=[X,ones(length(X),1)];

%train-validation-test split
[X_train,Y_train,X_check,Y_check] = train_test_split(X,Y,80);
[X_val,Y_val,X_test,Y_test] = train_test_split(X_check,Y_check,50);

X_train = [X_train,ones(length(X_train),1)];
X_val = [X_val,ones(length(X_val),1)];
X_test = [X_test,ones(length(X_test),1)];

alpha = 0.0000009;
labels=[1,2];

% Initialization of Weights
n = size(X);
n = n(2);
w = zeros(n,1);

%hyperparameter tuning
lambda = LambdaTune(X,Y,alpha,labels);
disp("Tuning Complete")
% [w,iter] = LogisticRegression(X,Y,w,lambda,alpha,10000000);


iter = 0;
    while true       
        iter = iter+1;
        grad = ((X'*(Y-Sigmoid(w,X)))+ lambda*w);
        w = w +alpha*grad;
        Norm=norm(grad)
        if norm(grad) <= 0.2
           
            break 
        end
        
    end
    Norm
   

disp("Training Complete")   

%% prediction on the test set
f_x = Sigmoid(w,X);
label = Classifier(f_x,1.5,labels)';
[X1,X2] = discriminant_Logistic(w);
accuracy = Acc(Y,label);
disp("Total number of iteration in Gradient Descent:"+ iter)
disp("Accuracy of Logistic Regression on Test Set: "+accuracy+"%" )

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
        lambda = lambda_array(i)
        
        [w,iter]=LogisticRegression(X,Y,w,lambda,alpha,10000);
        f_x = Sigmoid(w,X);
        label = Classifier(f_x,0.5,labels);
        accuracy = Acc(Y,label);
        if accuracy> acc
            acc = accuracy;
            lambda_final = lambda;
        end
        accuracy_array(i)= accuracy;
        
    end
    accuracy_array
    disp("Best Training Accuracy: "+acc+"%")
end

function [w,iter] = LogisticRegression(X,Y,w,lambda,alpha,max_iter)
    iter = 0;
    while iter<max_iter       
        iter = iter+1;
        grad = ((X'*(Y-Sigmoid(w,X)))+ lambda*w);
        w = w +alpha*grad;
        Norm=norm(grad);
        if norm(grad) <= 0.0001
           
            break 
        end
        
    end
    Norm
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
    xmin=1;
    xmax=length(X);
    n=n*xmax;
    n=int16(n);
    x=xmin+rand(1,n)*(xmax-xmin);
    x=int16(x);
    a=1;
    b=1;

    for i= 1:length(X)
        if ismember(i,x)
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
