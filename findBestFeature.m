function bestFeature = findBestFeature(VOCopts, fd, newdetector, currBestFeature,bbox)%, I)
bestBBox = [];

bestFeature = currBestFeature;
maxScore = newdetector.multiplier * sum(newdetector.w .* [currBestFeature 1]);


curHeight = VOCopts.firstdim * VOCopts.cellsize;
curWidth = VOCopts.seconddim * VOCopts.cellsize;

bbx1 = bbox(1);
bby1 = bbox(2);
bbx2 = bbox(3);
bby2 = bbox(4);
bbwidth = bbx2 - bbx1;
bbheight = bby2 - bby1;

for curScaleIndex = 1:length(fd)
    for i=1:VOCopts.rootsamples
        xlower = max(1,bbx1 - round(min(curWidth, bbwidth)/2));
        xupper = bbx2-round(min(bbwidth,curWidth)/2);
        ylower = max(1,bby1 - round(min(bbheight, curHeight)/2));
        yupper = bby2-round(min(bbheight,curHeight)/2);

        cury1 = round((ylower + ((yupper-ylower)*rand))/8)*8;
        curx1 = round((xlower + ((xupper-xlower)*rand))/8)*8;
        curx2 = curx1 + round(curWidth);
        cury2 = curx2 + round(curHeight);
        overlapPercent = calcMinOverlap(bbox,[curx1, cury1, curx2, cury2]);
        if (overlapPercent > 0.50)
            curCenter = [round((cury2 + cury1)/2); round((curx2+curx1)/2)];
            [~, curFeature] = pixelSpaceToHOGSpace(VOCopts, fd, curCenter, curScaleIndex);
            curScore = newdetector.multiplier * sum(newdetector.w .* [curFeature 1]);
            if (curScore > maxScore || size(bestBBox,1) == 0)
                maxScore = curScore;
                bestFeature = curFeature;
                bestBBox = [curx1; cury1; curx2; cury2];
            end
        end
    end

    curHeight = curHeight * (1/VOCopts.pyramidscale);
    curWidth = curWidth * (1/VOCopts.pyramidscale);
end   

function bestFeature = findBestFeatureAaron(VOCopts, fd, newdetector, bbox)%, I)
bestBBox = [];
bestFeature = zeros(VOCopts.firstdim* VOCopts.seconddim* VOCopts.blocksize^2 * VOCopts.numgradientdirections, 1);
maxScore = 0;

curHeight = VOCopts.firstdim * VOCopts.cellsize;
curWidth = VOCopts.seconddim * VOCopts.cellsize;

bbx1 = bbox(1);
bby1 = bbox(2);
bbx2 = bbox(3);
bby2 = bbox(4);
bbwidth = bbx2 - bbx1;
bbheight = bby2 - bby1;

for curScaleIndex = 1:length(fd)
    for curx1 = max(1,bbx1 - round(min(curWidth, bbwidth)/2)):8:bbx2-round(min(bbwidth,curWidth)/2)
        curx2 = curx1 + round(curWidth);
        for cury1 = max(1,bby1 - round(min(bbheight, curHeight)/2)):8:bby2-round(min(bbheight,curHeight)/2); 
            cury2 = cury1 + round(curHeight);
            overlapPercent = calcMinOverlap(bbox,[curx1, cury1, curx2, cury2]);
            if (overlapPercent > 0.50)
                curCenter = [round((cury2 + cury1)/2); round((curx2+curx1)/2)];
                [~, curFeature] = pixelSpaceToHOGSpace(VOCopts, fd, curCenter, curScaleIndex);
                curScore = newdetector.multiplier * sum(newdetector.w .* [curFeature 1]);
                if (curScore > maxScore || size(bestBBox,1) == 0)
                    maxScore = curScore;
                    bestFeature = curFeature;
                    bestBBox = [curx1; cury1; curx2; cury2];
                end
            end
        end
    end
    curHeight = curHeight * (1/VOCopts.pyramidscale);
    curWidth = curWidth * (1/VOCopts.pyramidscale);
end   