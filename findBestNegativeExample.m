function [maxScore,bestBBox, bestFeature] = findBestNegativeExample(VOCopts, fd, newdetector,...
    currBestFeature,bbox,numsamples)

bestBBox = [];

bestFeature = currBestFeature;
if length(bestFeature)>0
    maxScore = newdetector.multiplier * sum(newdetector.w .* [currBestFeature 1]);
else
    maxScore=1;
end


height = VOCopts.firstdim * VOCopts.cellsize;
width = VOCopts.seconddim * VOCopts.cellsize;

bbx1 = bbox(1);
bby1 = bbox(2);
bbx2 = bbox(3);
bby2 = bbox(4);
bbwidth = bbx2 - bbx1;
bbheight = bby2 - bby1;

for i=1:numsamples
    curScaleIndex = randsample(length(fd),1);
    curHeight = height * (1/VOCopts.pyramidscale)^(curScaleIndex-1);
    curWidth = width * (1/VOCopts.pyramidscale)^(curScaleIndex-1);
    xlower = max(1,bbx1 - round(min(curWidth, bbwidth)/2));
    xupper = bbx2-round(min(bbwidth,curWidth)/2);
    ylower = max(1,bby1 - round(min(bbheight, curHeight)/2));
    yupper = bby2-round(min(bbheight,curHeight)/2);

    cury1 = round((ylower + ((yupper-ylower)*rand))/8)*8;
    curx1 = round((xlower + ((xupper-xlower)*rand))/8)*8;
    curx2 = curx1 + round(curWidth);
    cury2 = curx2 + round(curHeight);
    curCenter = [round((cury2 + cury1)/2); round((curx2+curx1)/2)];
    [~, curFeature] = pixelSpaceToHOGSpace(VOCopts, fd, curCenter, curScaleIndex);
    curScore = newdetector.multiplier * sum(newdetector.w .* [curFeature 1]);
    if (curScore > maxScore || size(bestBBox,1) == 0)
        maxScore = curScore;
        bestFeature = curFeature;
        bestBBox = [curx1; cury1; curx2; cury2];
    end

end

