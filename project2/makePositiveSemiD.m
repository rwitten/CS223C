function [ sigma ] = makePositiveSemiD( row, col, n )
    sigma = zeros(row, col, n,n);

    for r = 1:row,
        for c = 1:col,
            b = randn(n,n);
            sigma(r,c,:,:) = b'*b;
        end
    end
end

