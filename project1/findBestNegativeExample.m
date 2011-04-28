function [maxScore, bestFeature, flipFeature] = findBestNegativeExample(VOCopts, fd, rootFilter)

maxScore = -inf;

bestRootLoc = zeros(2,1);
bestRootIndex = -1;

for curScaleIndex = (1+VOCopts.partstorootindexdiff):length(fd)
        %disp 'Filtering'
    %tic
    curHOG = squeeze(fd{curScaleIndex});
    curRootScores = HOGfilter(curHOG, rootFilter);

    [rootMaxScores lowys] = max(curRootScores);
    [curMaxScore lowx] = max(rootMaxScores);
    lowy = lowys(lowx);
    if (curMaxScore > maxScore || bestRootIndex == -1)
        maxScore = curMaxScore;
        bestRootLoc = [lowx-(VOCopts.seconddim-1); lowy-(VOCopts.firstdim-1)];
        bestRootIndex = curScaleIndex;
    end
end   

rootHog = padarray(fd{bestRootIndex},[VOCopts.firstdim-1, VOCopts.seconddim-1, 0]);
rootFeature = rootHog((bestRootLoc(2):(bestRootLoc(2)+VOCopts.firstdim-1)) + VOCopts.firstdim-1,...
    (bestRootLoc(1):(bestRootLoc(1)+VOCopts.seconddim-1)) + VOCopts.seconddim-1,:);
bestFeature = reshape(rootFeature, [1 numel(rootFeature)]);
flipFeature = reshape(rootFeature(:,end:-1:1,:), [1 numel(rootFeature)]);

end