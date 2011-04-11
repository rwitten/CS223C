%Image descriptor based on Histogram of Orientated Gradients for gray-level images. This code 
%was developed for the work: O. Ludwig, D. Delgado, V. Goncalves, and U. Nunes, 'Trainable 
%Classifier-Fusion Schemes: An Application To Pedestrian Detection,' In: 12th International IEEE 
%Conference On Intelligent Transportation Systems, 2009, St. Louis, 2009. V. 1. P. 432-437. In 
%case of publication with this code, please cite the paper above.


%Sources: Dalal+Trigg paper and Wikipedia

function H=HOG(VOCOpts, ImFull)
blocksize = VOCOpts.blocksize;
cellsize =  VOCOpts.cellsize;
numgradientdirections = VOCOpts.numgradientdirections;


Im = rgb2gray(ImFull);
hx = [-1,0, 1];
hy = -1*hx';

grad_xr = imfilter(double(Im),hx); %O. Ludwig, D. Delgado, V. Goncalves, and U. Nunes, 'Trainable 
                                   %Classifier-Fusion Schemes: An Application To Pedestrian Detection,'
                                   %gave me some wise guidance on how to do this.
                                   
grad_yu = imfilter(double(Im),hy);
<<<<<<< HEAD
angles=(atan(grad_yu,grad_xr)); %this is unsigned!
=======
>>>>>>> b8a33e1026d16aec9215a38b5f8d3c934e9a1ec3

hog = zeros(floor(size(Im,1)/cellsize)-2,floor(size(Im,2)/cellsize)-2, numgradientdirections*4);

for x = 2:floor(size(Im,1)/cellsize)-1,
    for y = 2:floor(size(Im,2)/cellsize)-1,
        Hupperright = extractHog(grad_xr,grad_yu, blocksize, cellsize, numgradientdirections, x,y,0,0);
        Hupperleft = extractHog(grad_xr,grad_yu, blocksize, cellsize, numgradientdirections, x,y,0,-1);
        Hlowerright = extractHog(grad_xr,grad_yu, blocksize, cellsize, numgradientdirections, x,y,-1,0);
        Hlowerleft = extractHog(grad_xr,grad_yu, blocksize, cellsize, numgradientdirections, x,y,-1,-1);
        hog(x-1,y-1, :) = [Hupperright;Hupperleft;Hlowerright;Hlowerleft];
    end
end

H=hog;




function vector=extractHog(grad_xr,grad_yu, blocksize, cellsize, numgradientdirections, x,y,xdirection,ydirection)
totalx = 0;
totaly = 0;
eps = 1e-4;
for i= 0+xdirection:1+xdirection,
    for j=0+ydirection:1+ydirection
        totalx = totalx+sum(sum(grad_xr((x+i-1)*cellsize+1:(x+i)*cellsize, (y+j-1)*cellsize+1:(y+j)*cellsize)));
        totaly = totaly+sum(sum(grad_yu((x+i-1)*cellsize+1:(x+i)*cellsize, (y+j-1)*cellsize+1:(y+j)*cellsize)));
    end
end

averageMag = sqrt(totalx^2  + totaly^2 + eps^2);

yblock = grad_yu( (x-1)*8+1:8*x, (y-1)*8+1:8*y);%/averageMag;
xblock = grad_xr( (x-1)*8+1:8*x, (y-1)*8+1:8*y);%/averageMag;
angles = pi - abs(mod(atan2(yblock, xblock),2*pi) - pi);


vector = bucketize(angles, yblock, xblock, numgradientdirections);

function vector= bucketize(angles, yblock, xblock, numgradientdirections);
eps = 1e-4;
vector = zeros(numgradientdirections,1);

for i=1:size(angles,1),
    for j = 1:size(angles,2),
        bucket = floor(max(0,(angles(i,j)-eps))*(9/(pi)))+1;
        vector(bucket) = vector(bucket) + sqrt(yblock(i,j)^2 + xblock(i,j)^2);
    end
end










