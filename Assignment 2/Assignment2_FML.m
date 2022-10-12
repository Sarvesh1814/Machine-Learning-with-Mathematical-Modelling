%Generating 20 real numbers for variable X from the uniform distribution

x= rand(20,1);
x=sort(x);

%Constructing Training set (x,y) using the given relation
epsilon = (0.25).*rand(1,1);
y=sin(2*pi*x)+epsilon;

%Constructing the Test Set
X_test= rand(50,1);
X_test=sort(X_test);
y_test=sin(2*pi*X_test)+epsilon;
%scatter(x,y,'r+'); %from the plot the relation between x & y is non-linear

%Therefore, we will use polynomial regression.


lambda_array = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7];%M=9

% for training 
a1 = [x,ones(length(x),1)]; %M=1
a2 = [x.^2,x,ones(length(x),1)]; %M=2
a3 = [x.^3,x.^2,x,ones(length(x),1)]; % M=3
a9 = [x.^9,x.^8,x.^7,x.^6,x.^5,x.^4,x.^3,x.^2,x,ones(length(x),1)];%M=9

% for testing 
a1_t = [X_test,ones(length(X_test),1)]; %M=1
a2_t = [X_test.^2,X_test,ones(length(X_test),1)];%M=2
a3_t = [X_test.^3,X_test.^2,X_test,ones(length(X_test),1)];%M=3
a9_t = [X_test.^9,X_test.^8,X_test.^7,X_test.^6,X_test.^5,X_test.^4,X_test.^3,X_test.^2,X_test,ones(length(X_test),1)];%M=9

i1 = eye(2);
i2 = eye(3);
i3 = eye(4);
i9 = eye(10);

lambda_array = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3,2^4,2^5,2^6,2^7];
RMSE1_array=[];
RMSE2_array=[];
RMSE3_array=[];
RMSE9_array=[];
min1=1;
min2=1;
min3=1;
min9=1;
for index = 1: 15
    
    lambda = lambda_array(index);
    
    u1 = inv(lambda*i1+a1'*a1)*a1'*y;
    u2 = inv(lambda*i2+a2'*a2)*a2'*y;
    u3 = inv(lambda*i3+a3'*a3)*a3'*y;
    u9 = inv(lambda*i9+a9'*a9)*a9'*y;

    y1 = a1*u1;
    y2 = a2*u2;
    y3 = a3*u3;
    y9 = a9*u9;
    
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
    RMSE_9 = (sum((y-y9).^2)/length(y)).^(0.5);
    RMSE9_array(index) = RMSE_9;
    if RMSE_9 < RMSE9_array(min9)
        min9 = index;
    end
    
end

% Final model M= 1

u1 = inv(lambda_array(min1)*i1+a1'*a1)*a1'*y;
%prediction on training set
y1 = a1*u1;
RMSE_1 = (sum((y-y1).^2)/length(y)).^(0.5);
%prediction on test set
yt1 = a1_t*u1;
RMSE_T1 = (sum((y_test-yt1).^2)/length(y_test)).^(0.5);

    
% Final model M= 2
u2 = inv(lambda_array(min2)*i2+a2'*a2)*a2'*y;
%prediction on training set
y2 = a2*u2;
RMSE_2 = (sum((y-y2).^2)/length(y)).^(0.5);
%prediction on test set
yt2 = a2_t*u2;
RMSE_T2 = (sum((y_test-yt2).^2)/length(y_test)).^(0.5);


% Final model M= 3
u3 = inv(lambda_array(min3)*i3+a3'*a3)*a3'*y;
%prediction on training set
y3 = a3*u3;
RMSE_3 = (sum((y-y3).^2)/length(y)).^(0.5);
%prediction on test set
yt3 = a3_t*u3;
RMSE_T3 = (sum((y_test-yt3).^2)/length(y_test)).^(0.5);

% Final model M= 9
u9 = inv(lambda_array(min9)*i9+a9'*a9)*a9'*y;
%prediction on training set
y9 = a9*u9;
RMSE_9 = (sum((y-y9).^2)/length(y)).^(0.5);
%prediction on test set
yt9 = a9_t*u9;
RMSE_T9 = (sum((y_test-yt9).^2)/length(y_test)).^(0.5);
