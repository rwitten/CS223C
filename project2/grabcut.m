function grabcut(im_name)
im_name='banana1.bmp';

im_data = imread(im_name);
%im_data = im_data(1:100,1:100,:);
% % display the image
% imagesc(im_data);
% 
% % a bounding box initialization
 disp('Draw a bounding box to specify the rough location of the foreground');
 set(gca,'Units','pixels');
 imshow(im_data)
 ginput(1);
 p1=get(gca,'CurrentPoint');fr=rbbox;p2=get(gca,'CurrentPoint');
 p=round([p1;p2]);
 xmin=min(p(:,1));xmax=max(p(:,1));
 ymin=min(p(:,2));ymax=max(p(:,2));
 [im_height, im_width, channel_num] = size(im_data);

 xmin = max(xmin, 1);
 xmax = min(im_width, xmax);
 ymin = max(ymin, 1);
 ymax = min(im_height, ymax);
 bbox = [xmin ymin xmax ymax];
%line(bbox([1 3 3 1 1]),bbox([2 2 4 4 2]),'Color',[1 0 0],'LineWidth',1);

%Paramaters for functions
params.K = 5;
params.numColors = channel_num;
params.numPixels= im_height * im_width;
params.height = im_height;
params.width = im_width;
params.numDirections = 8;
params.lambda = 50;
params.MaxIter = 1;
params.initIter = 4;

  %%Process Image Data
% % convert the pixel values to [0,1] for each R G B channel.
%true_im_data is for display
true_im_data = double(im_data) / 255;
back_im_data = true_im_data;
fore_im_data = true_im_data;
%Filter background image
h = 1/9 * ones(3,3); %fspecial('unsharp');
sharpGray = imfilter(double(rgb2gray(back_im_data)), h, 'replicate');
normIm = squeeze(.30 * back_im_data(:,:,1) + .59*back_im_data(:,:,2) + .11*back_im_data(:,:,3));
for i=1:params.numColors;
    back_im_data(:,:,i) = back_im_data(:,:,i) .* sharpGray./normIm;%imfilter(true_im_data(:,:,i), h, 'replicate');
end
true_im_data = reshape(true_im_data, [im_height*im_width 3]);
back_im_data = reshape(back_im_data,  [im_height*im_width 3]);
fore_im_data = reshape(fore_im_data,  [im_height*im_width 3]);
edge_im_data = true_im_data; %Temporary, can use a different image for weighting edges




%Create trimap, original alpha, and bounding box extractor array
trimap = ones(params.height,params.width);
trimap(ymin:ymax,xmin:xmax) = 3;
trimap = reshape(trimap, [params.numPixels 1]);
alpha = (trimap==3)+1;
params.unknownInd = alpha==2;

%initialize GMM components
% mu = rand(2, params.K, params.numColors);
% sigma = makePositiveSemiD(2, params.K, params.numColors);
% pi = 1/params.K * ones(2, params.K);
%im_data(logical(~params.unknownInd))

tic
%%Precompute the smoothing indices and weights
%Init matrices
indexMat = zeros(params.height, params.width, params.numDirections);
weightsMat = zeros(params.height, params.width, params.numDirections);
curDiffSq = zeros(params.height, params.width, params.numDirections);
shapedImage = reshape(edge_im_data, [params.height params.width params.numColors]);
padImage = padarray(shapedImage, [1 1 0]);
indexImage = padarray(reshape(1:params.numPixels, [params.height params.width]), [1 1]);
curIndex = 1;
for dy=-1:1
    for dx=-1:1
        if (dy ==0 && dx ==0) continue;
        end
        distFactor = 1/sqrt(dx^2 + dy^2);
        curDiffSq(:,:,curIndex) = sum((padImage(2:(end-1), 2:(end-1),:) - padImage((2+dy):(end-1+dy),(2+dx):(end-1+dx),:)).^2,3);
        indexMat(:,:,curIndex) = indexImage((2+dy):(end-1+dy),(2+dx):(end-1+dx));
        weightsMat(:,:,curIndex) = params.lambda * distFactor;
        curIndex = curIndex + 1;
    end
end
eps = 1e-6;
weightsMat(:,:,:) = weightsMat(:,:,:) .* exp(-1*bsxfun(@rdivide, curDiffSq, (eps + 2*mean(mean(mean(curDiffSq))))));
smoothIndices = reshape(indexMat, [params.numPixels, params.numDirections]);
smoothWeights = reshape(weightsMat, [params.numPixels, params.numDirections]);
clear weightsMat
clear indexMat
clear padImage
clear indexImage
clear curDiffSq
toc

% grabcut algorithm
fprintf('*************************\n');
fprintf('****grabcut algorithm****\n');
fprintf('*************************\n\n\n\n');

backmu = rand(params.K, params.numColors);
backSigma = squeeze(makePositiveSemiD(1, params.K, params.numColors));
backpi = rand(params.K, 1);
foremu = rand(params.K, params.numColors);
foreSigma = squeeze(makePositiveSemiD(1, params.K, params.numColors));
forePComponents = rand(params.K,1);

fprintf('lets reshape\n');
back_im_vec = reshape(back_im_data,  [im_height*im_width params.numColors]);
fore_im_vec = reshape(fore_im_data,  [im_height*im_width params.numColors]);


tic;
for iter=1:20%bs stopping criteria
    
    fprintf('number of foreground pixels %d\n',sum(alpha==2));
    fprintf('we are on iteration %d\n', iter);
    
%     fprintf('we are updating the cluster choices\n');
%     [ fgcluster,fg,bgcluster,bg ] = updateClusterChoices(params,alpha, im_data,...
%         mu, sigma, pi);
%     
%     fprintf('we are updating the cluster parameters\n');
%     [mu, sigma,pi] = updateClusterParameters(params, fgcluster,fg,bgcluster,bg);
%   
    %backStartStruct.mu = backGMFit.mu;
    %backStartStruct.Sigma = backGMFit.Sigma;
    %backStartStruct.PComponents = backGMFit.PComponents;
    %foreStartStruct.mu = foreGMFit.mu;
    %foreStartStruct.Sigma = foreGMFit.Sigma;
    %foreStartStruct.PComponents = foreGMFit.PComponents;

    backpixels = back_im_vec(logical(alpha==1),:);
    forepixels = fore_im_vec(logical(alpha==2),:);
    fprintf('done getting pixels\n');
    [backcluster] = assignCluster(params,backpixels,backmu,backSigma, ones(params.numColors,1));
    
    [forecluster] = assignCluster(params,forepixels,foremu,foreSigma, ones(params.numColors,1));
    %backGMFit = gmdistribution.fit(back_im_data(alpha==1,:), params.K, 'Options', gmmOptions, 'Start', backStartStruct);
    %foreGMFit = gmdistribution.fit(fore_im_data(alpha==2,:), params.K, 'Options', gmmOptions, 'Start', foreStartStruct);
    fprintf('done assigning clusters\n');
    [backmu, backSigma, backpi] = updateGaussian(params, backcluster, backpixels);
    [foremu, foreSigma, forepi] = updateGaussian(params, forecluster, forepixels);


%     fgallclusters=assigncluster(params, im_data, squeeze(mu(2,:,:)), squeeze(sigma(2,:,:,:)), squeeze(pi(2,:)));
%     bgallclusters=assigncluster(params, im_data, squeeze(mu(1,:,:)), squeeze(sigma(1,:,:,:)), squeeze(pi(1,:)));
%     [~,fgallcluster] = max(fgallclusters,[],2);
%     [~,bgallcluster] = max(bgallclusters,[],2);
    bgallcluster = assignCluster(params, back_im_vec,backmu, backSigma, backpi);
    fgallcluster = assignCluster(params, fore_im_vec,foremu, foreSigma, forepi);

    %fprintf('\n\n\n\n');
    %[alpha energy] = updateAlphaChoices(params, back_im_data, fore_im_data, mu, sigma,pi, fgallcluster, bgallcluster, smoothIndices, smoothWeights);
    %form_im_data = true_im_data;
    %form_im_data(logical(alpha==1),:) = 0;
    %form_im_data = reshape(form_im_data, [params.height params.width params.numColors]);
    %imshow(form_im_data);
    %drawnow;
end
toc

%drawClusters(fg,fgcluster);
figure();
disp_im_data = true_im_data;
disp_im_data(logical(alpha==1),:) = 0;
disp_im_data = reshape(disp_im_data, [params.height params.width params.numColors]);
imshow(disp_im_data);
% 
% 
