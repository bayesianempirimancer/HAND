function[ W ] = RevCor( Y,X,n )
    for i=1:n
        W(:,i) = circshift(Y,[1-i,0])'*X;
    end
    W=W/size(Y,1);
end
