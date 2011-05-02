function alpha = updateAlphaChoices(params, im_data, mu, sigma,pi,foreClusterIndices, backClusterIndices, smoothIndices, smoothWeights)

%Calculate weights of foreground and background
%Find elements in the equation
logpi = log(pi);
detSigma = zeros(size(sigma,1), size(sigma,2));
invSigma = zeros(size(sigma));
for i = 1:size(detSigma,1)
    for j = 1:size(detSigma,2)
        detSigma(i,j) = det(sigma(i,j,:,:));
        invSigma(i,j,:,:) = inv(sigma(i,j,:,:));
    end
end
logDetSigma = log(detSigma);

%Calculate weights
backWeights = -1*logpi(1,backClusterIndices) + 0.5 * logDetSigma(1,backClusterIndices) + ...
    0.5 * (im_data(:,:) - mu(1,backClusterIndices))' * invSigma(1, backClusterIndices,:,:) * ...
    (im_data(:,:) - mu(1,backClusterIndices));
foreWeights = -1*logpi(2,foreClusterIndices) + 0.5 * logDetSigma(2,foreClusterIndices) + ...
    0.5 * (im_data(:,:) - mu(2,foreClusterIndices))' * invSigma(2, foreClusterIndices,:,:) * ...
    (im_data(:,:) - mu(1,foreClusterIndices));

alpha = mexmaxflow(-1*backWeights, -1*foreWeights, -1*smoothIndices, -1*smoothWeights);
alpha = alpha+1;

end

