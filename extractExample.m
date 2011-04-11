function [ HOGVectors ] = extractExample( VOCopts, boundingbox,features )
%EXTRACTEXAMPLE Summary of this function goes here
%   Extract either a positive or a negative example from the image

xdim = size(features,1)*VOCopts.cellsize;
ydim = size(features,2)*VOCopts.cellsize;

if size(boundingbox,1)<1,
    centers = [];
    for i=1:5,
      centers = [centers [floor(rand()*xdim); floor(rand()*ydim)]];
    end
    
else
    centers = [];
    
    for i= 1:size(boundingbox,2),
        currbox = boundingbox(:,i);

        y1 = currbox(1);
        x1 = currbox(2);
        y2 = currbox(3);
        x2 = currbox(4);
        newcenter = [floor((x2 + x1)/2); floor((y2 + y1)/2)];
        offset = [(.5)*(rand-.5)*(x2-x1); .5*(rand-.5)*(y2-y1)];
        %newcenter = newcenter + offset;
        centers = [centers newcenter];
        
    end
end

HOGVectors = [];

for i=1:size(centers,2),
    [HOGCenter, HOGVector]=pixelSpaceToHOGSpace(VOCopts, features, centers(:,i));
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


