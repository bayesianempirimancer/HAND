function [Y] = WaveletConvolution(X, wavelet_type, K, prec);

%prec =2;
    [wpws, x] = wpfun(wavelet_type, K, prec);
    dswpws = wpws(2:end, :);%1:size(wpws, 2)/window_size:size(wpws, 2));
    dswpws = bsxfun(@rdivide, dswpws, sqrt(sum((dswpws').^2))');

    Y = nan(size(X, 1), K*size(X, 2));

    for i = 1:K
        Y(:, i:K:i+K*(size(X, 2)-1)) = conv2(dswpws(i, :)', 1, X, 'same');
    end

end

