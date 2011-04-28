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

grad_xr = imfilter(double(Im),hx,'same'); %O. Ludwig, D. Delgado, V. Goncalves, and U. Nunes, 'Trainable 
                                   %Classifier-Fusion Schemes: An Application To Pedestrian Detection,'
                                   %gave me some wise guidance on how to do this.
                                   
grad_yu = imfilter(double(Im),hy,'same');

%Populate orientaiton vectors for each cell
cellGrid = zeros(floor(size(Im,1)/cellsize), floor(size(Im,2)/cellsize), numgradientdirections);
for x = 1:floor(size(Im,1)/cellsize)
    for y = 1:floor(size(Im,2)/cellsize)
        yblock = grad_yu( max(1,((x-1)*cellsize+1 - cellsize/2)):min(size(grad_yu,1),(cellsize*x + cellsize/2)),...
            max(1,((y-1)*cellsize+1 - cellsize/2)):min(size(grad_yu,2),(cellsize*y + cellsize/2)));
        xblock = grad_xr( max(1,((x-1)*cellsize+1 - cellsize/2)):min(size(grad_xr,1),(cellsize*x + cellsize/2)),...
            max(1,((y-1)*cellsize+1 - cellsize/2)):min(size(grad_xr,2),(cellsize*y + cellsize/2)));
        angles = mod(atan(yblock./xblock),pi);
        angles1 = mod(angles + pi/(2*numgradientdirections), pi);
        angles2 = mod(angles - pi/(2*numgradientdirections), pi);
        cellGrid(x,y,:) = squeeze(cellGrid(x,y,:))' + bucketize(angles1, yblock, xblock, numgradientdirections)';
        cellGrid(x,y,:) = squeeze(cellGrid(x,y,:))' + bucketize(angles2, yblock, xblock, numgradientdirections)';
    end
end

%Normalize cells by block into features
hog = zeros((size(cellGrid,1)-2*(blocksize-1))*(size(cellGrid,2)-2*(blocksize-1)), numgradientdirections*blocksize*blocksize);
parfor i = 1:(size(cellGrid,1)-2*(blocksize-1))*(size(cellGrid,2)-2*(blocksize-1))
    [y x] = ind2sub([size(cellGrid,1) - 2*(blocksize-1), size(cellGrid,2) - 2*(blocksize-1)], i);
    curVec = zeros(blocksize*blocksize*numgradientdirections,1);
    for k=1:blocksize
        for j = 1:blocksize
            curBlock = cellGrid((y+blocksize-k):(y+2*blocksize-k - 1), (x + blocksize-j):(x+2*blocksize-j - 1), :);
            curBlock = min(hognormclip, curBlock./sqrt(sum(sum(sum(curBlock.^2))) + eps^2));
            curBlock = curBlock./sqrt(sum(sum(sum(curBlock.^2)))+eps^2);
            curVec((blocksize*(k-1) + j-1)*numgradientdirections+(1:numgradientdirections)) = curBlock(k,j,:);
        end
    end

    hog(i, :) = curVec;
end

H=reshape(hog, [size(cellGrid,1) - 2*(blocksize-1), size(cellGrid,2) - 2*(blocksize-1), size(hog,2)]);

function vector= bucketize(angles, yblock, xblock, numgradientdirections)
vector = zeros(numgradientdirections,1);
eps = 1e-6;

for i=1:size(angles,1),
    for j = 1:size(angles,2),
        bucket = floor(max(0,(angles(i,j)-eps))*(numgradientdirections/(pi)))+1;
        vector(bucket) = vector(bucket) + sqrt(yblock(i,j)^2 + xblock(i,j)^2);
    end
end










