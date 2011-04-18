function [HOGCenter, HOGVector] = pixelSpaceToHOGSpace(VOCopts, features, pixelcenter, pyramidIndex)
HOGCenter = round(pixelcenter.*(VOCopts.pyramidscale^(pyramidIndex-VOCopts.partstorootindexdiff - 1))/VOCopts.cellsize)-(VOCopts.blocksize-1);

firstlower = ceil(HOGCenter(1)-VOCopts.firstdim/2) + VOCopts.firstdim;
firstupper = ceil(HOGCenter(1)+VOCopts.firstdim/2-1) + VOCopts.firstdim;
secondlower = ceil(HOGCenter(2)-VOCopts.seconddim/2) + VOCopts.seconddim;
secondupper = ceil(HOGCenter(2)+VOCopts.seconddim/2-1) + VOCopts.seconddim;
featuresAtPyramidLevel = features{pyramidIndex};

paddedFeature = padArray(featuresAtPyramidLevel, [VOCopts.firstdim, VOCopts.seconddim, 0]);

HOGRegion = paddedFeature(firstlower:firstupper, secondlower:secondupper, :);
HOGVector = reshape(HOGRegion,[1,numel(HOGRegion)]);
  
end