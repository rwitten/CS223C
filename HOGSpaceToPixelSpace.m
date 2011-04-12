function [pixelBox, pixelCenter] = HOGSpaceToPixelSpace(VOCopts, HOGCenter, pyramidIndex)
scaleFactor =  ((1/VOCopts.pyramidscale)^(pyramidIndex-1));
pixelCenter = (HOGCenter+(VOCopts.blocksize-1))* VOCopts.cellsize * scaleFactor;
pixelBox = [pixelCenter(1) - scaleFactor * (VOCopts.firstdim*VOCopts.cellsize)/2; ...
    pixelCenter(1) + scaleFactor*(VOCopts.firstdim*VOCopts.cellsize)/2; ...
    pixelCenter(2) - scaleFactor *(VOCopts.seconddim*VOCopts.cellsize)/2; ...
    pixelCenter(2) + scaleFactor *(VOCopts.seconddim*VOCopts.cellsize)/2;];
end