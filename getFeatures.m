function [ fd ] = getFeatures( VOCopts, imageNumber )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
try
    % try to load features
    load(sprintf(VOCopts.exfdpath,imageNumber),'fd');
catch
    % compute and save features
    I=imread(sprintf(VOCopts.imgpath,imageNumber));
    fd=extractFD(VOCopts,I);
    save(sprintf(VOCopts.exfdpath,imageNumber),'fd');
end

end

