function [bestFeature flipFeature] = findBestFeature(VOCopts, fd, rootFilter, bbox)
bbx1 = bbox(1);
bby1 = bbox(2);
bbx2 = bbox(3);
bby2 = bbox(4);
bbWidth = bbx2 - bbx1;
bbHeight = bby2 - bby1;
cellsize = VOCopts.cellsize;

maxScore = -inf;

bestRootLoc = zeros(2,1);
bestRootIndex = -1;
maxSeenOverlap = 0;

for curScaleIndex = (1+VOCopts.partstorootindexdiff):length(fd)
    curScaleFactor = 1/(VOCopts.pyramidscale^(curScaleIndex-VOCopts.partstorootindexdiff-1));
    hogToPixel = cellsize * curScaleFactor;
    %disp 'Filtering'
    %tic
    curHOG = squeeze(fd{curScaleIndex});
     
    minrootx = max(1,ceil((bbx1 - hogToPixel)/hogToPixel) - ceil(min(bbWidth/hogToPixel,VOCopts.seconddim)/2));
    maxrootx = min(size(curHOG,2),ceil((bbx2 - hogToPixel)/hogToPixel) + ceil(min(bbWidth/hogToPixel,VOCopts.seconddim)/2));
    minrooty = max(1,ceil((bby1 - hogToPixel)/hogToPixel) - ceil(min(bbHeight/hogToPixel,VOCopts.firstdim)/2));
    maxrooty = min(size(curHOG,1),ceil((bby2 - hogToPixel)/hogToPixel) + ceil(min(bbHeight/hogToPixel,VOCopts.firstdim)/2));
    
    curRootScores = HOGfilter(curHOG(minrooty:maxrooty,minrootx:maxrootx), rootFilter);
    
    for cury1 = 1:VOCopts.rootskip:size(curRootScores,1)
        truey = (cury1 + minrooty-1) - (VOCopts.firstdim-1);
        truey2 = truey + VOCopts.firstdim;
        pixely1 = round((truey-1)*hogToPixel + 1 + hogToPixel);
        pixely2 = round((truey2-1)*hogToPixel + 1 + hogToPixel);
        for curx1 =  1:VOCopts.rootskip:size(curRootScores,2)
            truex = (curx1 + minrootx-1) - (VOCopts.seconddim-1);
            truex2 = truex + VOCopts.seconddim;
            pixelx1 = round((truex-1)*hogToPixel + 1 + hogToPixel);
            pixelx2 = round((truex2-1)*hogToPixel + 1 + hogToPixel);
            minOverlap = calcMinOverlap(bbox, [pixelx1 pixely1 pixelx2 pixely2]);
            if (minOverlap > 0.50 || minOverlap >= maxSeenOverlap)
                curMaxScore = curRootScores(cury1,curx1);
                if (curMaxScore > maxScore || bestRootIndex == -1 || (minOverlap <= 0.5 && minOverlap >= maxSeenOverlap))
                    maxScore = curMaxScore;
                    bestRootLoc = [truex; truey];
                    bestRootIndex = curScaleIndex;
                    maxSeenOverlap = minOverlap;
                end
            end
        end
    end
end

rootHog = padarray(fd{bestRootIndex},[VOCopts.firstdim-1, VOCopts.seconddim-1, 0]);
rootFeature = rootHog((bestRootLoc(2):(bestRootLoc(2)+VOCopts.firstdim-1)) + VOCopts.firstdim-1,...
    (bestRootLoc(1):(bestRootLoc(1)+VOCopts.seconddim-1)) + VOCopts.seconddim-1,:);
bestFeature = reshape(rootFeature, [1 numel(rootFeature)]);
flipFeature = reshape(rootFeature(:,end:-1:1,:), [1 numel(rootFeature)]);
end