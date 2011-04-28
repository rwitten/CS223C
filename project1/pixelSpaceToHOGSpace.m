function [HOGCenter, HOGVector] = pixelSpaceToHOGSpace(VOCopts, features, pixelcenter, pyramidIndex)
HOGCenter = ceil(pixelcenter/VOCopts.cellsize)-(VOCopts.blocksize-1);
HOGCenter = round(HOGCenter * (VOCopts.pyramidscale^(pyramidIndex-VOCopts.partstorootindexdiff-1)));

firstlower = ceil(HOGCenter(1)-VOCopts.firstdim/2);
firstupper = ceil(HOGCenter(1)+VOCopts.firstdim/2-1);
secondlower = ceil(HOGCenter(2)-VOCopts.seconddim/2);
secondupper = ceil(HOGCenter(2)+VOCopts.seconddim/2-1);
featuresAtPyramidLevel = features{pyramidIndex};

if firstlower < 1,
    firstlower = 1;
    firstupper = min(size(featuresAtPyramidLevel,1),VOCopts.firstdim);
end

if firstupper > size(featuresAtPyramidLevel,1),
    firstlower = max(1,size(featuresAtPyramidLevel,1)-VOCopts.firstdim+1);
    firstupper = size(featuresAtPyramidLevel,1);
end

if secondlower < 1,
    secondlower = 1;
    secondupper = min(size(featuresAtPyramidLevel,2),VOCopts.seconddim);
end

if secondupper > size(featuresAtPyramidLevel,2),
    secondlower = max(1,size(featuresAtPyramidLevel,2)-VOCopts.seconddim+1);
    secondupper = size(featuresAtPyramidLevel,2);
end

HOGRegion = featuresAtPyramidLevel(firstlower:firstupper, secondlower:secondupper, :);
HOGVector = reshape(HOGRegion,[1, prod(size(HOGRegion))]);
  
end