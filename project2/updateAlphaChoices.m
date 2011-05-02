function alpha = updateAlphaChoices(params, im_data, mu, sigma,pi,foreClusterIndices, backClusterIndices, smoothIndices, smoothWeights)

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
logDetSigma = log(detSigma)

%Calculate weights
% backWeights = -1*squeeze(logpi(1,backClusterIndices)) + 0.5 * squeeze(logDetSigma(1,backClusterIndices)) + ...
%     0.5 * (im_data(:,:) - squeeze(mu(1,backClusterIndices,:))') * squeeze(invSigma(1, backClusterIndices,:,:)) * ...
%     (im_data(:,:) - squeeze(mu(1,backClusterIndices,:))')';
% foreWeights = -1*squeeze(logpi(2,foreClusterIndices)) + 0.5 * squeeze(logDetSigma(2,foreClusterIndices)) + ...
%     0.5 * (im_data(:,:) - squeeze(mu(2,foreClusterIndices,:))') * squeeze(invSigma(2, foreClusterIndices,:,:)) * ...
%     (im_data(:,:) - squeeze(mu(2,foreClusterIndices,:))')
backWeights = zeros(params.numPixels,1);
foreWeights = zeros(params.numPixels,1);
for i = 1:params.K
    curBackInd = backClusterIndices == i;
    curForeInd = foreClusterIndices == i;
    backPixelDiff = bsxfun(@minus, squeeze(im_data(curBackInd,:)), squeeze(mu(1,i,:))');
    forePixelDiff = bsxfun(@minus, squeeze(im_data(curForeInd,:)), squeeze(mu(2,i,:))');
    backWeights(curBackInd) = -1*squeeze(logpi(1,i)) + 0.5 * squeeze(logDetSigma(1,i)) + ...
        0.5 * sum(backPixelDiff * squeeze(invSigma(1, i,:,:)) .* backPixelDiff,2);
    foreWeights(curForeInd) = -1*squeeze(logpi(2,i)) + 0.5 * squeeze(logDetSigma(2,i)) + ...
        0.5 * sum(forePixelDiff * squeeze(invSigma(2, i,:,:)) .* forePixelDiff,2);
end
% for i=1:params.numPixels;
%     curClusInd = backClusterIndices(i);
%     backWeights(i) = -1*squeeze(logpi(1,curClusInd)) + 0.5 * squeeze(logDetSigma(1,curClusInd)) + ...
%         0.5 * (squeeze(im_data(i,:)) - squeeze(mu(1,curClusInd,:))') * squeeze(invSigma(1, curClusInd,:,:)) * ...
%         (squeeze(im_data(i,:)) - squeeze(mu(1,curClusInd,:))')';
%     curClusInd = foreClusterIndices(i);
%     foreWeights(i) = -1*squeeze(logpi(2,curClusInd)) + 0.5 * squeeze(logDetSigma(2,curClusInd)) + ...
%         0.5 * (squeeze(im_data(i,:)) - squeeze(mu(2,curClusInd,:))') * squeeze(invSigma(2, curClusInd,:,:)) * ...
%         (squeeze(im_data(i,:)) - squeeze(mu(2,curClusInd,:))')';
% end
backWeights;

alpha = mexmaxflow(backWeights, foreWeights, smoothIndices, smoothWeights);
alpha = alpha+1;

end

