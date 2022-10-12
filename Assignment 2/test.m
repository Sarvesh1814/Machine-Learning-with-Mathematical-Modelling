
%Reading Dataset
motorcycle = readmatrix("motorcycle.csv");
x1= motorcycle(:,1);
x2= motorcycle(:,2);

X = [x1,x2];
d= [1/sqrt(2),1/sqrt(2)];

u = sum(X)/length(X);
X_U = X-u;
var_X = (d*X_U'*X_U*d')/(length(X)-1)
scatter(x1,x2,"black")
hold on;
scatter(x1-u(1),x2-u(2),"red")
legend("Actual","Mean Sperated")
