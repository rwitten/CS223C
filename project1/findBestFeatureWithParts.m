function [bestFeature flipFeature newpbboxes] = findBestFeatureWithParts(VOCopts, fd, rootFilter, partFilters,...
            deformScores, pbboxes, bbox)
bbx1 = bbox(1);
bby1 = bbox(2);
bbx2 = bbox(3);
bby2 = bbox(4);
bbWidth = bbx2 - bbx1;
bbHeight = bby2 - bby1;
cellsize = VOCopts.cellsize;

bestFeature = [];
maxScore = -inf;
numparts = VOCopts.numparts;

bestRootLoc = zeros(2,1);
bestPartLocs = zeros(2,numparts);
bestRelLocs = zeros(2,numparts);
partLocs = zeros(2,numparts);
relLocs = zeros(2,numparts);
bestRootIndex = -1;
maxSeenOverlap = 0;

for curScaleIndex = (1+VOCopts.partstorootindexdiff):length(fd)
    curScaleFactor = 1/(VOCopts.pyramidscale^(curScaleIndex-VOCopts.partstorootindexdiff-1));
    hogToPixel = cellsize * curScaleFactor;
    %disp 'Filtering'
    %tic
    curHOG = fd{curScaleIndex};
    curPartHOG = fd{curScaleIndex-VOCopts.partstorootindexdiff};
    curRootScores = HOGfilter(curHOG, rootFilter);
    curPartScores = {};
    curPartXGrid = {};
    curPartYGrid = {};
    for i=1:VOCopts.numparts
        curPartScores{i} = HOGfilter(curPartHOG, partFilters{i});
        [curPartXGrid{i} curPartYGrid{i}] = meshgrid((1:size(curPartScores{i},2))-(size(partFilters{i},2)-1),...
            (1:size(curPartScores{i},1)) - (size(partFilters{i},1)-1));
    end
    %toc
    %Scan root
    %disp 'Finding max'
    %tic
    minrootx = max(1,ceil((bbx1 - hogToPixel)/hogToPixel) - ceil(min(bbWidth/hogToPixel,VOCopts.seconddim)/2) + VOCopts.seconddim-1);
    maxrootx = min(size(curRootScores,2),ceil((bbx2 - hogToPixel)/hogToPixel) + ceil(min(bbWidth/hogToPixel,VOCopts.seconddim)/2) + VOCopts.seconddim-1);
    minrooty = max(1,ceil((bby1 - hogToPixel)/hogToPixel) - ceil(min(bbHeight/hogToPixel,VOCopts.firstdim)/2) + VOCopts.firstdim-1);
    maxrooty = min(size(curRootScores,1),ceil((bby2 - hogToPixel)/hogToPixel) + ceil(min(bbHeight/hogToPixel,VOCopts.firstdim)/2) + VOCopts.firstdim-1);
    for cury1 = minrooty:VOCopts.rootskip:maxrooty
        truey = cury1 - (VOCopts.firstdim-1);
        pspacetruey = 2*truey-1;
        truey2 = truey + VOCopts.firstdim;
        pixely1 = round((truey-1)*hogToPixel + 1 + hogToPixel);
        pixely2 = round((truey2-1)*hogToPixel + 1 + hogToPixel);
        for curx1 =  minrootx:VOCopts.rootskip:maxrootx
            truex = curx1 - (VOCopts.seconddim-1);
            pspacetruex = 2*truex-1;
            truex2 = truex + VOCopts.seconddim;
            pixelx1 = round((truex-1)*hogToPixel + 1 + hogToPixel);
            pixelx2 = round((truex2-1)*hogToPixel + 1 + hogToPixel);
            minOverlap = calcMinOverlap(bbox, [pixelx1 pixely1 pixelx2 pixely2]);
            if (minOverlap > 0.50 || minOverlap >= maxSeenOverlap)
                curMaxScore = curRootScores(cury1,curx1);
                for i=1:VOCopts.numparts          
                    curPartWidth = squeeze(pbboxes(3,i) - pbboxes(1,i) + 1);
                    curPartHeight = squeeze(pbboxes(4,i) - pbboxes(2,i) + 1);
                    pspacey = pspacetruey + curPartHeight-1;
                    pspacex = pspacetruex + curPartWidth-1;
                    minpy = floor(max(1,pspacey -  curPartHeight/2));
                    maxpy = ceil(min(size(curPartXGrid{i},1), minpy + 2*VOCopts.firstdim + curPartHeight +1));
                    minpx = floor(max(1,pspacex -  curPartWidth/2));
                    maxpx = ceil(min(size(curPartXGrid{i},2), minpx + 2*VOCopts.seconddim + curPartWidth +1));             
                    curDistX = abs(curPartXGrid{i}(minpy:VOCopts.partskip:maxpy, minpx:VOCopts.partskip:maxpx) - (pspacetruex + pbboxes(1,i)));
                    curDistY = abs(curPartYGrid{i}(minpy:VOCopts.partskip:maxpy, minpx:VOCopts.partskip:maxpx) - (pspacetruey + pbboxes(2,i)));
                    distScore = (deformScores(1,i) * curDistX + deformScores(3,i) * curDistX.^2)/(2*VOCopts.seconddim) + ...
                         (deformScores(2,i) * curDistY + deformScores(4,i) * curDistY.^2)/(2*VOCopts.firstdim);
                    curTotalPartScore = curPartScores{i}(minpy:VOCopts.partskip:maxpy, minpx:VOCopts.partskip:maxpx) + distScore;
                    [partMaxScores lowys] = max(curTotalPartScore);
                    [partMax lowx] = max(partMaxScores);
                    lowy = lowys(lowx);
                    partLocs(:,i) = [curPartXGrid{i}(lowy + minpy-1,lowx + minpx-1); curPartYGrid{i}(lowy+minpy-1,lowx+minpx-1)];
                    relLocs(:,i) = [curDistX(lowy,lowx); curDistY(lowy,lowx);];
                    curMaxScore = curMaxScore + partMax;
                end
                if (curMaxScore > maxScore || bestRootIndex == -1 || (minOverlap <= 0.5 && minOverlap >= maxSeenOverlap))
                    maxScore = curMaxScore;
                    bestRootLoc = [truex; truey];
                    bestPartLocs = partLocs;
                    bestRelLocs = relLocs;
                    bestRootIndex = curScaleIndex;
                    maxSeenOverlap = minOverlap;
                end
            end
        end
    end
    %toc
end

rootHog = padarray(fd{bestRootIndex},[VOCopts.firstdim-1, VOCopts.seconddim-1, 0]);
rootFeature = rootHog((bestRootLoc(2):(bestRootLoc(2)+VOCopts.firstdim-1)) + VOCopts.firstdim-1,...
    (bestRootLoc(1):(bestRootLoc(1)+VOCopts.seconddim-1)) + VOCopts.seconddim-1,:);
bestFeature = reshape(rootFeature, [1 numel(rootFeature)]);
newpbboxes = zeros(size(pbboxes));
flipFeature = reshape(rootFeature(:,end:-1:1,:), [1 numel(rootFeature)]);
for i=1:VOCopts.numparts
    curPartWidth = round(pbboxes(3,i) - pbboxes(1,i) + 1);
    curPartHeight = round(pbboxes(4,i) - pbboxes(2,i) + 1);
    partHog = padarray(fd{bestRootIndex-VOCopts.partstorootindexdiff}, [curPartHeight-1, curPartWidth-1, 0]);
    partFeature = partHog((bestPartLocs(2,i):(bestPartLocs(2,i)+curPartHeight-1)) + curPartHeight-1, ...
        (bestPartLocs(1,i):(bestPartLocs(1,i) + curPartWidth - 1)) + curPartWidth-1,:);
    bestFeature = [bestFeature reshape(partFeature, [1 numel(partFeature)])];
    flipFeature = [flipFeature reshape(partFeature(:,end:-1:1,:), [1 numel(partFeature)])];
    newpbboxes(:,i) = [bestRelLocs(1,i); bestRelLocs(2,i); bestRelLocs(1,i)+curPartWidth-1; bestRelLocs(2,i)+curPartHeight-1];
end

for i=1:VOCopts.numparts
    bestFeature = [bestFeature bestRelLocs(:,i)' (bestRelLocs(:,i).^2)'];
    flipFeature = [flipFeature bestRelLocs(:,i)' (bestRelLocs(:,i).^2)'];
end

end