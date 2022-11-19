clc;
clear all;
close all;

%% case 1

mu_A = [-1 1];
Sigma_A = [0.6 0; 0.0 0.6];
X_A = mvnrnd(mu_A,Sigma_A,2000);

mu_B = [1 1];
Sigma_B = [0.6 0; 0.0 0.6];
X_B = mvnrnd(mu_B,Sigma_B,2000);

figure;
scatter(X_A(:,1),X_A(:,2),10,"red","filled")
hold on;
scatter(X_B(:,1),X_B(:,2),10,"blue","filled")
hold on;
[X1,X2]= discriminant(X_A,mu_A,mu_B);
plot(X1,X2,"Black")
axis([-4 4 -4 4])
legend("Class: Dataset A","Class Dataset B")
title("case1")



%% case 2

mu_A = [-1 1];
Sigma_A = [0.7 0; 0.0 0.3];
X_A = mvnrnd(mu_A,Sigma_A,2000);


mu_B = [2 2];
Sigma_B = [0.7 0; 0.0 0.3];
X_B = mvnrnd(mu_B,Sigma_B,2000);

[X3,X4] = discriminant2(X_A,mu_A,mu_B,Sigma_B);

figure;
scatter(X_A(:,1),X_A(:,2),10,"red","filled")
hold on;
scatter(X_B(:,1),X_B(:,2),10,"blue","filled")
hold on;
plot(X3,X4,"Black")
axis([-4 4 -4 4])
legend("Class: Dataset A","Class Dataset B")
title("case2")


%% case 3

mu_A = [-1 1];
sigma_A = [0.6 0.25; 0.25 0.4];
X_A = mvnrnd(mu_A,sigma_A,2000);

mu_B = [2 2];
sigma_B = [0.7 0.35; 0.35 0.7];
X_B = mvnrnd(mu_B,sigma_B,2000);

[X5,X6] = discriminant3(mu_A,mu_B,sigma_A,sigma_B,X_A,X_B);

figure;
scatter(X_A(:,1),X_A(:,2),10,"red","filled")
hold on;
scatter(X_B(:,1),X_B(:,2),10,"blue","filled")
hold on;
% X= discriminant3(mu_A,mu_B,Sigma_A,Sigma_B);
scatter(X5,X6,"Black","filled")
axis([-4 4 -4 4])
legend("Class: Dataset A","Class Dataset B")
title("case3")

%% Definition
function [X1,X2] = discriminant(X_A,mu_A,mu_B)
    d = ((mu_A'-mu_B')'*(mu_A'+mu_B'))/2;
    m = (mu_A'-mu_B')';
     
    if m(2) == 0
        X1 =zeros(100,1);
        X2 = linspace(min(min(X_A)),max(max(X_A)));
    else
        X1= linspace(min(X_A),max(X_A));
        X2 = ((d*ones(100,1)-(X1*m(1))))/m(2);
    end
end

function [X1,X2] = discriminant2(X_A,mu_A,mu_B,sigma)
    d = (inv(sigma)*(mu_A'-mu_B'))'*(mu_A'+mu_B')*0.5;
    m = (inv(sigma)*(mu_A'-mu_B'))'

    if m(2) == 0
        X1 = zeros(100,1);
        X2 = linspace(min(min(X_A)),max(max(X_A)));
    else
        X1= linspace(min(min(X_A)),max(max(X_A)));
        X2 = ((d*ones(100,1)-(X1*m(1))'))/m(2);
    end
end

function [X5,X6] = discriminant3(mu_A,mu_B,sigma_A,sigma_B,X_A,X_B)

    
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
