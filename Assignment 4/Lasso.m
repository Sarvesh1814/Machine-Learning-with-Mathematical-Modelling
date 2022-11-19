%%  Lasso Regression

% alpha = 0.002;
% iter = 0;
% grad_norm = 1;
% 
% [lambda,M] = regularized_fit(X_train,Y_train);
% u= rand(M+1,1);
% while grad_norm >0.01
%     
%     A1 = Poly_AugMat(X_train,M);
%     iter=iter+1;
%     grad= ((lambda*sign(u))-(A1'*(Y_train-(A1*u))));
%     u = u - (alpha*grad);
%     grad_norm = norm(grad)
%     norm_array(iter) = grad_norm;
%     no(iter)= iter;
%     
% end
% 
% 
% Y_pred = A1*u;
% 
% figure;
% scatter(X_train,Y_pred,"red*")
% hold on;
% scatter(X_train,Y_train,"black+")
% legend("Prediction on training Data","Actual Training data")
% 
% figure;
% A1_t = Poly_AugMat(X_test,M);
% Y_pred_t = A1_t*u;
% scatter(X_test,Y_pred_t,"red*")
% hold on;
% scatter(X_test,Y_test,"black+")
% legend("Prediction on Test Data","Actual Test data")
% 
% figure;
% plot(no,norm_array)
% Xlabel("No of Iterations")
% Ylabel("Norm of Gradient")
% title("Cost VS Iterations")
