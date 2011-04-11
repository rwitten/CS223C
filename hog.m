%Image descriptor based on Histogram of Orientated Gradients for gray-level images. This code 
%was developed for the work: O. Ludwig, D. Delgado, V. Goncalves, and U. Nunes, 'Trainable 
%Classifier-Fusion Schemes: An Application To Pedestrian Detection,' In: 12th International IEEE 
%Conference On Intelligent Transportation Systems, 2009, St. Louis, 2009. V. 1. P. 432-437. In 
%case of publication with this code, please cite the paper above.


%Sources: Dalal+Trigg paper and Wikipedia

function H=HOG(VOCOpts, ImColor)
blocksize = VOCOpts.blocksize;
cellsize =  VOCOpts.cellsize;
numgradientdirections = VOCOpts.numgradientdirections;
hognormclip = VOCOpts.hognormclip;
eps = 1e-6;


Im = rgb2gray(ImColor);
hx = [-1,0, 1];
hy = hx';

grad_xr = imfilter(double(Im),hx); %O. Ludwig, D. Delgado, V. Goncalves, and U. Nunes, 'Trainable 
                                   %Classifier-Fusion Schemes: An Application To Pedestrian Detection,'
                                   %gave me some wise guidance on how to do this.
                                   
grad_yu = imfilter(double(Im),hy);
<<<<<<< HEAD
angles=(atan(grad_yu,grad_xr)); %this is unsigned!
=======
>>>>>>> b8a33e1026d16aec9215a38b5f8d3c934e9a1ec3

%Populate orientaiton vectors for each cell
cellGrid = zeros(floor(size(Im,1)/cellsize), floor(size(Im,2)/cellsize),numgradientdirections);
for x = 1:floor(size(Im,1)/cellsize)
    for y = 1:floor(size(Im,2)/cellsize)
        yblock = grad_yu( ((x-1)*8+1):(8*x), ((y-1)*8+1):(8*y));
        xblock = grad_xr( ((x-1)*8+1):(8*x), ((y-1)*8+1):(8*y));
        angles = mod(atan(yblock./xblock),pi);
        cellGrid(x,y,:) = bucketize(angles, yblock, xblock, numgradientdirections);
    end
end

%Normalize cells by block into features
hog = zeros(size(cellGrid,1)-2, size(cellGrid,2)-2, numgradientdirections*4);
for x = 2:size(cellGrid,1)-1
    for y = 2:size(cellGrid,2)-1
        Hupperright = squeeze(min(hognormclip, cellGrid(x,y,:) ./ sqrt(sum(sum(sum(cellGrid((x-1):x, y:(y+1), :).^2))) + eps^2)));
        Hupperright = Hupperright ./ sqrt(norm(Hupperright,2)^2 + eps^2);
        Hupperleft = squeeze(min(hognormclip, cellGrid(x,y,:) ./ sqrt(sum(sum(sum(cellGrid((x-1):x,(y-1):y, :).^2))) + eps^2)));
        Hupperleft = Hupperleft ./ sqrt(norm(Hupperleft,2)^2 + eps^2);
        Hlowerright = squeeze(min(hognormclip, cellGrid(x,y,:) ./ sqrt(sum(sum(sum(cellGrid(x:(x+1), y:(y+1), :).^2))) + eps^2)));
        Hlowerright = Hlowerright./ sqrt(norm(Hlowerright,2)^2 + eps^2);
        Hlowerleft = squeeze(min(hognormclip, cellGrid(x,y,:) ./ sqrt(sum(sum(sum(cellGrid(x:(x+1), (y-1):y, :).^2))) + eps^2)));
        Hlowerleft = Hlowerleft ./ sqrt(norm(Hlowerleft,2)^2 + eps^2);
        hog(x-1,y-1, :) = [Hupperright;Hupperleft;Hlowerright;Hlowerleft];
    end
end

H=hog;

function vector= bucketize(angles, yblock, xblock, numgradientdirections);
vector = zeros(numgradientdirections,1);
eps = 1e-6;

for i=1:size(angles,1),
    for j = 1:size(angles,2),
        bucket = floor(max(0,(angles(i,j)-eps))*(numgradientdirections/(pi)))+1;
        vector(bucket) = vector(bucket) + sqrt(yblock(i,j)^2 + xblock(i,j)^2);
    end
end










