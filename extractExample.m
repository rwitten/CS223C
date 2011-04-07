function [ HOGVector ] = extractExample( VOCopts, boundingbox,features )
%EXTRACTEXAMPLE Summary of this function goes here
%   Extract either a positive or a negative example from the image

xdim = size(features,1)*VOCopts.cellsize;
ydim = size(features,2)*VOCopts.cellsize;

if size(boundingbox,1)<1,
    example = []; %IN THE FUTURE GENERATE A NEGATIVE EXAMPLE
    center = [floor(rand()*xdim); floor(rand()*ydim)];
else,
    currbox = boundingbox(:,max(1, end-1));

    y1 = currbox(1);
    x1 = currbox(2);
    y2 = currbox(3);
    x2 = currbox(4);

    center = [floor((x2 + x1)/2); floor((y2 + y1)/2)];
end


[HOGCenter, HOGVector]=pixelSpaceToHOGSpace(VOCopts, features, center);

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

function [HOGCenter, HOGVector] = pixelSpaceToHOGSpace(VOCopts, features, pixelcenter)
HOGCenter = ceil(pixelcenter/8)-1;

firstlower = ceil(HOGCenter(1)-VOCopts.firstdim/2);
firstupper = ceil(HOGCenter(1)+VOCopts.firstdim/2-1);
secondlower = ceil(HOGCenter(2)-VOCopts.seconddim/2);
secondupper = ceil(HOGCenter(2)+VOCopts.seconddim/2-1);

if size(features,1) < VOCopts.firstdim || size(features,2) < VOCopts.seconddim,
    HOGCenter = [];
    HOGVector = [];
    return;
end

if firstlower < 1,
    firstlower = 1;
    firstupper = VOCopts.firstdim;
end

if firstupper > size(features,1),
    firstlower = size(features,1)-VOCopts.firstdim+1;
    firstupper = size(features,1);
end

if secondlower < 1,
    secondlower = 1;
    secondupper = VOCopts.seconddim;
end

if secondupper > size(features,2),
    secondlower = size(features,2)-VOCopts.seconddim+1;
    secondupper = size(features,2);
end

HOGVector = reshape(features(firstlower:firstupper, secondlower:secondupper, :)...
    ,[1, prod(size(features(firstlower:firstupper, secondlower:secondupper, :)))]);
  
end

function [pixelBox] = HOGSpaceToPixelSpace(VOCopts, features, HOGCenter)
pixelCenter = (HOGCenter+1)* VOCopts.cellsize;
pixelBox = [pixelCenter(1) - (VOCopts.firstdim*VOCopts.cellsize)/2; ...
    pixelCenter(1) + (VOCopts.firstdim*VOCopts.cellsize)/2; ...
    pixelCenter(2) - (VOCopts.seconddim*VOCopts.cellsize)/2; ...
    pixelCenter(2) + (VOCopts.seconddim*VOCopts.cellsize)/2;];

    
end