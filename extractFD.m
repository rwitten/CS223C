function fd = extractfd(VOCopts,I)
minfirstdim = (VOCopts.firstdim+2*(VOCopts.blocksize-1)) * VOCopts.cellsize;
minseconddim = (VOCopts.seconddim+2*(VOCopts.blocksize-1)) * VOCopts.cellsize;
curScale = 1;
curI = I;
fd = {};
while (size(curI,1) >= minfirstdim && size(curI,2) >= minseconddim)
    curFd = HOG(VOCopts, curI);
    fd = [fd curFd];
    curScale = curScale * VOCopts.pyramidscale;
    curI = imresize(I, curScale, 'bilinear'); 
end