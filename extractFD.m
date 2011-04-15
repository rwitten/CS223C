function fd = extractfd(VOCopts,I)
minfirstdim = (VOCopts.firstdim+2*(VOCopts.blocksize-1)) * VOCopts.cellsize;
minseconddim = (VOCopts.seconddim+2*(VOCopts.blocksize-1)) * VOCopts.cellsize;
curScale = 2;
fd = {};
curI = imresize(I, curScale, 'bilinear'); 
while (size(curI,1) >= minfirstdim && size(curI,2) >= minseconddim)
    curFd = hog(VOCopts, curI);
    fd{end+1} = curFd;
    curScale = curScale * VOCopts.pyramidscale;
    curI = imresize(I, curScale, 'bilinear'); 
end
fd;