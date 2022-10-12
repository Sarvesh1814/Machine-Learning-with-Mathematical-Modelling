%% Reading the data and Initialization
motorcycle = readmatrix("motorcycle.csv");
X= motorcycle(:,1);
Y= motorcycle(:,2);
counter =1;
RMSE10=[];
u=[];
lambda= 0.0078; % from the previous question
i9 = eye(10);
a=rand(10,1);
b=rand(10,1);
a=a';
b=b';
%% Leave One Out Algorithm Code 
for index = 1:length(X)
    c=1;
    x=[];
    X_test=[];
    y=[];
    y_test=[];
    for index2 = 1:length(X)
        if index2 ~= counter 
            x(c) = X(index2);
            y(c)=  Y(index2);
            c=c+1;
        
        else
            X_test(1) = X(index2);
            y_test(1)= Y(index2);

        end
    end
    x=x';
    y=y';
    A10 = AugMaT(x,10,a,b);
    identity = eye(11);
    u_temp = Weights(lambda,A10,y,10);
    At10 = AugMaT(X_test,10,a,b);
    y_pred10 = At10*u_temp;
    RMSE10(counter)= (((y_test-y_pred10).^2).^0.5);
    MAE10(counter) =sum(y_test-y_pred10)/length(y_test);
    counter=counter+1;
    
end

Mean_RMSE= sum(RMSE10)/length(RMSE10);
Std_RMSE = ((sum(RMSE10 - Mean_RMSE).^2)/length(RMSE10)).^0.5;

Mean_MAE = sum(MAE10)/length(MAE10);
Std_MAE  = (((sum(MAE10 - Mean_MAE)).^2)/length(MAE10)).^0.5;

%% User written Functions

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

function u = Weights(lambda,A,Y,n)
    identity = eye(n+1);
    u = inv(lambda*identity+A'*A)*A'*Y;
end



