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