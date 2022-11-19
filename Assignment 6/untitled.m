% Kernel Matrix
function h= K_Mat(d_i,d_j,sigma)

    for i = 1:length(d_i)
        for j = 1:length(d_j)
            norm_vector = abs(d_i(i,:) - d_j(j,:));
            norm_value = ((sum(norm_vector.^2)).^0.5);
            h(i,j)= exp(-(norm_value/(sigma.^2))); %RBF Kernel
        end
        h(i,(length(d_j)+1))= 1;
    end
end


% Kernal Weights
function V = Kernel_weights(lambda,H,y)
    V = inv(lambda.*eye(101)+H'*H)*H'*y;
end

