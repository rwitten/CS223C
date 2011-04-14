function [pfilters bboxes] = initializePartsFromRoot(VOCopts, rootFilter)
partfirstdim = VOCopts.partfirstdim;
partseconddim = VOCopts.partseconddim;
numparts = VOCopts.numparts;
rootfirstdim = VOCopts.firstdim;
rootseconddim = VOCopts.seconddim;

shapedFilter = reshape(rootFilter(1:end-1), [rootfirstdim rootseconddim VOCopts.blocksize^2*VOCopts.numgradientdirections]);
vertRep = repmat([1 0], 1, size(shapedFilter,1));
vertRep = imfilter(vertRep, ones(1, 2*size(shapedFilter,1)), 'full');
vertRep = vertRep(1:2*size(shapedFilter,1));
horRep = repmat([1 0], 1, size(shapedFilter,2));
horRep = imfilter(horRep, ones(1, 2*size(shapedFilter,2)), 'full');
horRep = horRep(1:2*size(shapedFilter,1));
doubledFilter = shapedFilter(vertRep, horRep, :);

%figure();
%visualizeHOG(shapedFilter);

energyMap = zeros(2*rootfirstdim, 2*rootseconddim);
for y=1:2*rootfirstdim
    for x=1:2*rootseconddim
        energyMap(y,x) = norm(squeeze(doubledFilter(y,x,:).*(doubledFilter(y,x,:)>0)),2);
    end
end
%figure();
%imagesc(255*energyMap./max(max(energyMap)));

sameOrient = ones(partfirstdim, partseconddim);
flipOrient = sameOrient';

bboxes = zeros(4,numparts);
pfilters = zeros(partfirstdim*partseconddim*VOCopts.blocksize^2*VOCopts.numgradientdirections, numparts);
for i=1:numparts
    sameMap = imfilter(energyMap, sameOrient, 'full');
    flipMap = imfilter(energyMap, flipOrient, 'full');
    maxSame = max(max(sameMap));
    maxFlip = max(max(flipMap));
    if (maxSame > maxFlip)
        curMap = sameMap;
        curOrient = sameOrient;
    else
        curMap = flipMap;
        curOrient = flipOrient;
    end
    [maxvals,lowys] = max(curMap);
    [~, lowx] = max(maxvals);
    lowy = lowys(lowx);
    lowx = lowx - (size(curOrient,2)-1);
    lowy = lowy - (size(curOrient,1)-1);
    highx = lowx + (size(curOrient,2)-1);
    highy = lowy + (size(curOrient,1)-1);
    
    %zero out energy map
    energyMap(max(1,lowy):min(end,highy),max(1,lowx):min(end,highx)) = 0;
    %figure();
    %imagesc(255*energyMap./max(max(energyMap)));

    
    %Convert coordinates to be in part space
    bboxes(:,i) = [lowx; lowy; highx; highy;];
    
    %createPart       
    paddedFilter = padarray(doubledFilter, [size(curOrient) 0]);
    filterArea = paddedFilter((size(curOrient,1)-1) + (lowy:highy), (size(curOrient,2)-1) + (lowx:highx),:);
    pfilters(:,i) = reshape(filterArea, [1 prod(size(filterArea))]);
end