function [alpha energy] = updateAlphaChoices(params, back_im_data, fore_im_data, backmu, backSigma, backpi, foremu, foreSigma, forepi, foreClusterIndices, backClusterIndices, smoothIndices, smoothWeights)
%Calculate weights of foreground and background
%Find elements in the equation
backlogpi = log(backpi);
forelogpi = log(forepi);
backDetSigma = zeros(size(backSigma,1),1);
foreDetSigma = zeros(size(foreSigma,1),1);
backInvSigma = zeros(size(backSigma));
foreInvSigma = zeros(size(foreSigma));
for i = 1:size(backDetSigma,1)
    backDetSigma(i) = det(squeeze(backSigma(i,:,:)));
    backInvSigma(i,:,:) = inv(squeeze(backSigma(i,:,:)));
end
for i = 1:size(foreDetSigma,1)
    foreDetSigma(i) = det(squeeze(foreSigma(i,:,:)));
    foreInvSigma(i,:,:) = inv(squeeze(foreSigma(i,:,:)));
end
backLogDetSigma = log(backDetSigma)';
foreLogDetSigma = log(foreDetSigma)';


backWeights = zeros(params.numPixels,1);
foreWeights = zeros(params.numPixels,1);
for i = 1:params.backK
    curBackInd = backClusterIndices == i;
    backPixelDiff = bsxfun(@minus, squeeze(back_im_data(curBackInd,:)), squeeze(backmu(i,:)));
    backWeights(curBackInd) = -1*squeeze(backlogpi(i)) + 0.5 * squeeze(backLogDetSigma(i)) + ...
        0.5 * sum(backPixelDiff * squeeze(backInvSigma(i,:,:)) .* backPixelDiff,2);
end
for i = 1:params.foreK
    curForeInd = foreClusterIndices == i;
    forePixelDiff = bsxfun(@minus, squeeze(fore_im_data(curForeInd,:)), squeeze(foremu(i,:)));
    foreWeights(curForeInd) = -1*squeeze(forelogpi(i)) + 0.5 * squeeze(foreLogDetSigma(i)) + ...
        0.5 * sum(forePixelDiff * squeeze(foreInvSigma(i,:,:)) .* forePixelDiff,2);
end

% backWeights(~params.unknownInd) = -1e13;
% foreWeights(~params.unknownInd) = 1e13;

[alpha energy] = mexmaxflow(-1*backWeights, -1*foreWeights, smoothIndices, -1*smoothWeights);
alpha = alpha+1;
alpha(~params.unknownInd) = 1;
energy = energy * -1;

end

