
function [ HOGVectors ] = extractExample( VOCopts, boundingbox,features )
%EXTRACTEXAMPLE Summary of this function goes here
%   Extract either a positive or a negative example from the image

xdim = (size(features{1},1))*VOCopts.cellsize;
ydim = (size(features{1},2))*VOCopts.cellsize;

if size(boundingbox,1)<1,
    centers = [];
    pyramidIndices = [];
    for i=1:5
      newPyramidIndex = floor(rand()*length(features) + 1);
      scale = VOCopts.pyramidscale ^ (newPyramidIndex-1);
      scaleWidth = VOCopts.cellsize*VOCopts.seconddim/scale;
      scaleHeight = VOCopts.cellsize*VOCopts.firstdim/scale;
      centers = [centers [floor(rand()*(ydim - scaleHeight) + scaleHeight/2); floor(rand()*(xdim-scaleWidth) +scaleWidth/2)]];
      pyramidIndices = [pyramidIndices newPyramidIndex];
    end
    
else
    centers = [];
    pyramidIndices = [];
    
    for i= 1:size(boundingbox,2),
        currbox = boundingbox(:,i);

        x1 = currbox(1);
        y1 = currbox(2);
        x2 = currbox(3);
        y2 = currbox(4);
        newcenter = [floor((y2 + y1)/2); floor((x2 + x1)/2)];
        yScale = log2(abs(y2-y1)/(VOCopts.firstdim*VOCopts.cellsize))/log2(1/VOCopts.pyramidscale);
        xScale = log2(abs(x2-x1)/(VOCopts.seconddim*VOCopts.cellsize))/log2(1/VOCopts.pyramidscale);
        scaleIndex = min(length(features)-1,max(0,round((xScale + yScale)/2))) + 1;
        
        %Check bounding box for legality
        newwidth = VOCopts.cellsize*VOCopts.seconddim * ((1/VOCopts.pyramidscale)^(scaleIndex-1));
        newheight = VOCopts.cellsize*VOCopts.firstdim * ((1/VOCopts.pyramidscale)^(scaleIndex-1));
        newbox = round([newcenter(2) - newwidth/2, newcenter(1) - newheight/2, newcenter(2) + newwidth/2, newcenter(1) + newheight/2]);
        overlapPercent = calcMinOverlap(currbox,newbox);
        
        if (overlapPercent < 0.55) continue;
        end
        
        %newcenter = newcenter + offset;
        centers = [centers newcenter];
        pyramidIndices = [pyramidIndices scaleIndex];
        %figure();
        %drawBoundingBox(VOCopts, I, currbox, scaleIndex);
        
    end
end

HOGVectors = [];

for i=1:size(centers,2),
    [HOGCenter, HOGVector]=pixelSpaceToHOGSpace(VOCopts, features, centers(:,i), pyramidIndices(i));
    HOGVectors = [HOGVectors; HOGVector];
end

%This code helps you draw boxes.
%for i = 1:xdim,
%    for j = 1:ydim,
%        if i < x2 && i>x1 && j < y2 && j >y1,
%            Im(i,j,3) = 1e4;
%        end
%    end    
%end

%imwrite(Im, 'boom.png', 'png');
%example = [x2-x1; y2-y1];


end


