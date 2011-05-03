function [alpha energy] = updateAlphaChoices(params, back_im_data, fore_im_data, backmu, backSigma, backpi, foremu, foreSigma, forepi, foreClusterIndices, backClusterIndices, smoothIndices, smoothWeights)

sigma(1,:,:,:) = backSigma;
sigma(2,:,:,:) = foreSigma;
mu(1,:,:) = backmu;
mu(2,:,:) = foremu;
pi(1,:) = backpi;
pi(2,:) = forepi

%Calculate weights of foreground and background
%Find elements in the equation
logpi = log(pi);
detSigma = zeros(size(sigma,1), size(sigma,2));
invSigma = zeros(size(sigma));
for i = 1:size(detSigma,1)
    for j = 1:size(detSigma,2)
        detSigma(i,j) = det(squeeze(sigma(i,j,:,:)));
        invSigma(i,j,:,:) = inv(squeeze(sigma(i,j,:,:)));
    end
end
logDetSigma = log(detSigma);


backWeights = zeros(params.numPixels,1);
foreWeights = zeros(params.numPixels,1);
for i = 1:params.K
    curBackInd = backClusterIndices == i;
    curForeInd = foreClusterIndices == i;
    backPixelDiff = bsxfun(@minus, squeeze(back_im_data(curBackInd,:)), squeeze(mu(1,i,:))');
    forePixelDiff = bsxfun(@minus, squeeze(fore_im_data(curForeInd,:)), squeeze(mu(2,i,:))');
    backWeights(curBackInd) = -1*squeeze(logpi(1,i)) + 0.5 * squeeze(logDetSigma(1,i)) + ...
        0.5 * sum(backPixelDiff * squeeze(invSigma(1, i,:,:)) .* backPixelDiff,2);
    foreWeights(curForeInd) = -1*squeeze(logpi(2,i)) + 0.5 * squeeze(logDetSigma(2,i)) + ...
        0.5 * sum(forePixelDiff * squeeze(invSigma(2, i,:,:)) .* forePixelDiff,2);
end

backWeights(~params.unknownInd) = -1e6;
foreWeights(~params.unknownInd) = 1e6;

[alpha energy] = mexmaxflow(-1*backWeights, -1*foreWeights, smoothIndices, -1*smoothWeights);
alpha = alpha+1;
alpha(~params.unknownInd) = 1;
energy = energy * -1;

end

