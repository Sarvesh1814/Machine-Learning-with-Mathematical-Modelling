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


% Least Square Polynomial Regression Model of order M=1,2,3 and 9

% Model 1 (M=1)

%for training set
a1 = [x,ones(length(x),1)];
u1 = inv((a1'*a1))*a1'*y;
y_pred1_train = a1*u1;
%plotting results for prediction train case prediction

scatter(x,y,'r+');
hold on;
scatter(x,y_pred1_train,'b+');
legend("Actual","Predicted")

%for test set
a1_t = [X_test,ones(length(X_test),1)];
y_pred1 = a1_t*u1;

%plotting results for prediction test case prediction

scatter(X_test,y_test,'r+');
hold on;
scatter(X_test,y_pred1,'b+');
legend("Actual","Predicted")

%Root Mean Square Error
RM_T1= y-y_pred1_train;
RM_T1 = abs(RM_T1);
RM_T1 = RM_T1.^2;
RMS_T1= sum(RM_T1);
RMS_T1= RMS_T1/length(x);
RMSE_T1 = RMS_T1.^(1/2);

RM_1= y_test-y_pred1;
RM_1 = abs(RM_1);
RM_1 = RM_1.^2;
RMS_1= sum(RM_1);
RMS_1= RMS_1/length(X_test);
RMSE_1 = RMS_1.^(1/2);


% Model 2 (M=2)

%for training set
a2 = [x.^2,x,ones(length(x),1)];
u2 = inv((a2'*a2))*a2'*y;
y_pred2_train = a2*u2;

%plotting results for prediction train case prediction

scatter(x,y,'r+');
hold on;
scatter(x,y_pred2_train,'b+');
legend("Actual","Predicted")

%for test set
a2_t = [X_test.^2,X_test,ones(length(X_test),1)];
y_pred2 = a2_t*u2;

%plotting results for prediction test case prediction
scatter(X_test,y_test,'r+');
hold on;
scatter(X_test,y_pred2,'b+');
legend("Actual","Predicted")

%Root Mean Square Error
RM_T2= y-y_pred2_train;
RM_T2 = abs(RM_T2);
RM_T2 = RM_T2.^2;
RMS_T2= sum(RM_T2);
RMS_T2= RMS_T2/length(x);
RMSE_T2 = RMS_T2.^(1/2);


RM_2= y_test-y_pred2;
RM_2 = abs(RM_2);
RM_2 = RM_2.^2;
RMS_2= sum(RM_2);
RMS_2= RMS_2/length(X_test);
RMSE_2 = RMS_2.^(1/2);



% Model 3 (M=3)

%for training set
a3 = [x.^3,x.^2,x,ones(length(x),1)];
u3 = inv((a3'*a3))*a3'*y;
y_pred3_train = a3*u3;

%plotting results for prediction train case prediction

scatter(x,y,'r+');
hold on;
scatter(x,y_pred3_train,'b+');
legend("Actual","Predicted")

%for test set
a3_t = [X_test.^3,X_test.^2,X_test,ones(length(X_test),1)];
y_pred3 = a3_t*u3;

%plotting results for prediction test case prediction

scatter(X_test,y_test,'r+');
hold on;
scatter(X_test,y_pred3,'b+');
legend("Actual","Predicted")
%Root Mean Square Error
RM_T3= y-y_pred3_train;
RM_T3 = abs(RM_T3);
RM_T3 = RM_T3.^2;
RMS_T3= sum(RM_T3);
RMS_T3= RMS_T3/length(x);
RMSE_T3 = RMS_T3.^(1/2);


RM_3= y_test-y_pred3;
RM_3 = abs(RM_3);
RM_3 = RM_3.^2;
RMS_3= sum(RM_3);
RMS_3= RMS_3/length(X_test);
RMSE_3 = RMS_3.^(1/2);





% Model 9 (M=9)

%for training set
a9 = [x.^9,x.^8,x.^7,x.^6,x.^5,x.^4,x.^3,x.^2,x,ones(length(x),1)];
u9 = inv((a9'*a9))*a9'*y;
y_pred9_train = a9*u9;

%plotting results for prediction train case prediction

scatter(x,y,'r+');
hold on;
scatter(x,y_pred9_train,'b+');
legend("Actual","Predicted")

%for test set
a9_t = [X_test.^9,X_test.^8,X_test.^7,X_test.^6,X_test.^5,X_test.^4,X_test.^3,X_test.^2,X_test,ones(length(X_test),1)];
y_pred9 = a9_t*u9;

%plotting results for prediction test case prediction

scatter(X_test,y_test,'r+');
hold on;
scatter(X_test,y_pred9,'b*');
legend("Actual","Predicted")
%Root Mean Square Error
RM_T9= y-y_pred9_train;
RM_T9 = abs(RM_T9);
RM_T9 = RM_T9.^2;
RMS_T9= sum(RM_T9);
RMS_T9= RMS_T9/length(x);
RMSE_T9 = RMS_T9.^(1/2);


RM_9= y_test-y_pred9;
RM_9 = abs(RM_9);
RM_9 = RM_9.^2;
RMS_9= sum(RM_9);
RMS_9= RMS_9/length(X_test);
RMSE_9 = RMS_9.^(1/2);


%Common plotting for training

Y_by_x = sin(2*pi*x);
plot(x,Y_by_x,'k-');
hold on;
plot(x,y,"red--")
hold on;
scatter(x,y_pred1_train,'blue*');
hold on;
scatter(x,y_pred2_train,'green+');
hold on;
scatter(x,y_pred3_train,"red")
hold on;
scatter(x,y_pred9_train,"black");
xlabel("X train")
ylabel("Y Train/Y Pred")
legend("sin(2*pi*x)","Y Train","M= 1","M=2","M= 3","M= 9")

%Common plotting for training

Y_by_Xt = sin(2*pi*X_test);
plot(X_test,Y_by_Xt,'k-');
hold on;
plot(X_test,y_test,"red--")
hold on;
scatter(X_test,y_pred1,'blue*');
hold on;
scatter(X_test,y_pred2,'green+');
hold on;
scatter(X_test,y_pred3,"red")
hold on;
scatter(X_test,y_pred9,"black");
xlabel("X test")
ylabel("Y Test/Y Pred")
legend("sin(2*pi*Xtest)","Y Test","M= 1","M=2","M= 3","M= 9")