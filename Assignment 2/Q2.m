% Bivariate Data 
%% Generating 20 real numbers for variable x1 and x2 from the uniform distribution
x1 = rand(20,1);
x2 = rand(20,1); 
x1=sort(x1);
x2 = sort(x2);
epsilon = (0.25).*rand(1,1);

%% for plotting purpose
[X1,X2] = meshgrid(x1,x2);
Y =sin(2*pi*(x1.^2+x2.^2).^(0.5))+epsilon;



%% Constructing Training set (x,y) using the given relation
epsilon = (0.25).*rand(1,1);
y=sin(2*pi*(x1.^2+x2.^2).^(0.5))+epsilon;

%Constructing the Test Set
X_test1= rand(50,1);
X_test2= rand(50,1);
X_test1=sort(X_test1);
X_test2= sort(X_test2);

y_test=sin(2*pi*(X_test1.^2+ X_test2.^2).^(0.5))+epsilon;

[XT1,XT2] = meshgrid(X_test1,X_test2);
YT1 =sin(2*pi*(XT1.^2+ XT2.^2).^(0.5))+epsilon;
 

%from the plot the relation between x & y is non-linear

%Therefore, we will use polynomial regression.

% Range of values for the value of Lambda (Regularization)
lambda_array = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7];

% for training 
a1 = [x1,x2,ones(length(x1),1)]; %M=1
a2 = [x1.^2,x2.^2,x1.*x2,x1,x2,ones(length(x1),1)]; %M=2
a3 = [x1.^5,x2.^5,x1.^4.*x2,x1.^3.*x2.^2,x1.^2.*x2.^3,x1.*x2.^4,x1.^4,x2.^4,x1.^3.*x2,x1.^2.*x2.^2,x1.*x2.^3,x1.^3,x2.^3,x1.^2.*x2,x1.*x2.^2,x1.^2,x2.^2,x1.*x2,x1,x2,ones(length(x1),1)]; % M=5


% for testing 
a1_t = [X_test1,X_test2,ones(length(X_test1),1)]; %M=1
a2_t = [X_test1.^2,X_test2.^2,X_test1.*X_test2,X_test1,X_test2,ones(length(X_test1),1)]; %M=2
a3_t = [X_test1.^5,X_test2.^5,X_test1.^4.*X_test2,X_test1.^3.*X_test2.^2,X_test1.^2.*X_test2.^3,X_test1.*X_test2.^4,X_test1.^4,X_test2.^4,X_test1.^3.*X_test2,X_test1.^2.*X_test2.^2,X_test1.*X_test2.^3,X_test1.^3,X_test2.^3,X_test1.^2.*X_test2,X_test1.*X_test2.^2,X_test1.^2,X_test2.^2,X_test1.*X_test2,X_test1,X_test2,ones(length(X_test1),1)];

i1 = eye(3);
i2 = eye(6);
i3 = eye(21);


lambda_array = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7];
RMSE1_array=[];
RMSE2_array=[];
RMSE3_array=[];

% storing the best value of lambda for M= 1,2,5
min1=1;
min2=1;
min3=1;

for index = 1: 15
    
    lambda = lambda_array(index);
    
    u1 = inv(lambda*i1+a1'*a1)*a1'*y;
    u2 = inv(lambda*i2+a2'*a2)*a2'*y;
    u3 = inv(lambda*i3+a3'*a3)*a3'*y;
   

    y1 = a1*u1;
    y2 = a2*u2;
    y3 = a3*u3;
   
    RMSE_1 = (sum((y-y1).^2)/length(y)).^(0.5);
    RMSE1_array(index) = RMSE_1;
    if RMSE_1< RMSE1_array(min1)
        min1 = index;
    end
    RMSE_2 = (sum((y-y2).^2)/length(y)).^(0.5);
    RMSE2_array(index) = RMSE_2;
    if RMSE_2< RMSE2_array(min2)
        min2 = index;
    end

    RMSE_3 = (sum((y-y3).^2)/length(y)).^(0.5);
    RMSE3_array(index) = RMSE_3;
    if RMSE_3< RMSE3_array(min3)
        min3 = index;
    end
    
    
end

%% Final model M= 1

u1 = inv(lambda_array(min1)*i1+a1'*a1)*a1'*y;
%prediction on training set
y1 = a1*u1;
RMSE_1 = (sum((y-y1).^2)/length(y)).^(0.5);

%prediction on test set
yt1 = a1_t*u1;
RMSE_T1 = (sum((y_test-yt1).^2)/length(y_test)).^(0.5);

figure;
Z= u1(1)*X1 + u1(2)*X2 + u1(3);
surf(X1,X2,Z);
hold on;
scatter3(x1,x2,y,'r*');
hold on;
scatter3(X_test1,X_test2,y_test,'blackx');
legend("Predicted plane","TrainingData","TestData")
title("For Training Data and Test Data (M=1)")
xlabel("X1")
ylabel("X2")
zlabel("Y")




    
%% Final model M= 2
u2 = inv(lambda_array(min2)*i2+a2'*a2)*a2'*y;
%prediction on training set
y2 = a2*u2;
RMSE_2 = (sum((y-y2).^2)/length(y)).^(0.5);

%prediction on test set
yt2 = a2_t*u2;
RMSE_T2 = (sum((y_test-yt2).^2)/length(y_test)).^(0.5);

figure;
Z2= u2(1)*X1.^2 + u2(2)*X2.^2 + u2(3)*X1.*X2+u2(4)*X1+u2(5)*X2+u2(6);
surf(X1,X2,Z2);
hold on;
scatter3(x1,x2,y,'r*');
hold on;
scatter3(X_test1,X_test2,y_test,'blackx');
legend("Predicted plane","TrainingData","TestData")
title("For Training Data and Test Data (M=2)")
xlabel("X1")
ylabel("X2")
zlabel("Y")






%% Final model M= 5
u3 = inv(lambda_array(min3)*i3+a3'*a3)*a3'*y;
%prediction on training set
y3 = a3*u3;
RMSE_3 = (sum((y-y3).^2)/length(y)).^(0.5);

%prediction on test set
yt3 = a3_t*u3;
RMSE_T3 = (sum((y_test-yt3).^2)/length(y_test)).^(0.5);

figure;
Z3= u3(1)*X1.^5+u3(2)*X2.^5+u3(3)*X1.^4.*X2+u3(4)*X1.^3.*X2.^2+u3(5)*X1.^2.*X2.^3+u3(6)*X1.*X2.^4+u3(7)*X1.^4+u3(8)*X2.^4+u3(9)*X1.^3.*X2+u3(10)*X1.^2.*X2.^2+u3(11)*X1.*X2.^3+u3(12)*X1.^3+u3(13)*X2.^3+u3(14)*X1.^2.*X2+u3(15)*X1.*X2.^2+u3(16)*X1.^2+u3(17)*X2.^2+u3(18)*X1.*X2+u3(19)*X1+u3(20)*X2+u3(21);
surf(X1,X2,Z3);
hold on;
scatter3(x1,x2,y,'r*');
hold on;
scatter3(X_test1,X_test2,y_test,'blackx');
legend("Predicted plane","TrainingData","TestData")
title("For Training Data and Test Data (M=5)")
xlabel("X1")
ylabel("X2")
zlabel("Y")










