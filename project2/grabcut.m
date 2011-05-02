function grabcut(im_name)
im_name='fullmoon.bmp';

im_data = imread('flower.jpg');
%im_data = im_data(1:100,1:100,:);
% % display the image
% imagesc(im_data);
% 
% % a bounding box initialization
%  disp('Draw a bounding box to specify the rough location of the foreground');
%  set(gca,'Units','pixels');
%  imshow(im_data)
%  ginput(1);
%  p1=get(gca,'CurrentPoint');fr=rbbox;p2=get(gca,'CurrentPoint');
%  p=round([p1;p2]);
%  xmin=min(p(:,1));xmax=max(p(:,1));
%  ymin=min(p(:,2));ymax=max(p(:,2));
  [im_height, im_width, channel_num] = size(im_data);

% % convert the pixel values to [0,1] for each R G B channel.
im_data = reshape(double(im_data) / 255, [im_height*im_width 3]);
%  
%  xmin = max(xmin, 1);
%  xmax = min(im_width, xmax);
%  ymin = max(ymin, 1);
%  ymax = min(im_height, ymax);

 xmin = 150;
 xmax = 270;
 ymin = 116;
 ymax = 235;

bbox = [xmin ymin xmax ymax];
%line(bbox([1 3 3 1 1]),bbox([2 2 4 4 2]),'Color',[1 0 0],'LineWidth',1);

if channel_num ~= 3
    disp('This image does not have all the RGB channels, you do not need to work on it.');
    return;
end

params.K = 2;
params.numColors = channel_num;
params.numPixels= im_height * im_width;
params.height = im_height;
params.width = im_width;
params.numDirections = 8;
params.lambda = 50;

trimap = ones(params.height,params.width);
tic
trimap(ymin:ymax,xmin:xmax) = 3;
toc
trimap = reshape(trimap, [params.numPixels 1]);

alpha = (trimap==3)+1;
params.unknownInd = alpha==2;

%initialize GMM components
% mu = rand(2, params.K, params.numColors);
% sigma = makePositiveSemiD(2, params.K, params.numColors);
% pi = 1/params.K * ones(2, params.K);
%im_data(logical(~params.unknownInd))
backGMFit = gmdistribution.fit(im_data(params.unknownInd==0,:), params.K);
foreGMFit = gmdistribution.fit(im_data(params.unknownInd==1,:), params.K);
mu(1,:,:) = backGMFit.mu;
mu(2,:,:) = foreGMFit.mu;
sigma(1,:,:,:) = permute(backGMFit.Sigma, [3 1 2]);
sigma(2,:,:,:) = permute(foreGMFit.Sigma, [3 1 2]);
pi(1,:) = backGMFit.PComponents;
pi(2,:) = foreGMFit.PComponents;

tic
%%Precompute the smoothing indices and weights
%Calculate beta
%form_im_data(1,:,:) = im_data;
%pixel_mat = repmat(form_im_data, [params.numPixels 1 1]);
%pixel_diff_sq = (pixel_mat - permute(pixel_mat, [2 1 3])).^2;
%beta = 3;%1/(2*mean(mean(sum(pixel_diff_sq, 3))));
indexMat = zeros(params.height, params.width, params.numDirections);
weightsMat = zeros(params.height, params.width, params.numDirections);
curDiffSq = zeros(params.height, params.width, params.numDirections);
shapedImage = reshape(im_data, [params.height params.width params.numColors]);
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
weightsMat(:,:,:) = weightsMat(:,:,:) .* exp(-1*bsxfun(@rdivide, curDiffSq, (eps + 2*mean(curDiffSq,3))));
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

tic;
for iter=1:20%bs stopping criteria
    sum(alpha==2)
    fprintf('we are on iteration %d\n', iter);
    
%     fprintf('we are updating the cluster choices\n');
%     [ fgcluster,fg,bgcluster,bg ] = updateClusterChoices(params,alpha, im_data,...
%         mu, sigma, pi);
%     
%     fprintf('we are updating the cluster parameters\n');
%     [mu, sigma,pi] = updateClusterParameters(params, fgcluster,fg,bgcluster,bg);
%     
    backGMFit = gmdistribution.fit(im_data(~params.unknownInd,:), params.K);
    foreGMFit = gmdistribution.fit(im_data(params.unknownInd,:), params.K);
    mu(1,:,:) = backGMFit.mu;
    mu(2,:,:) = foreGMFit.mu;
    sigma(1,:,:,:) = permute(backGMFit.Sigma, [3 1 2]);
    sigma(2,:,:,:) = permute(foreGMFit.Sigma, [3 1 2]);
    pi(1,:) = backGMFit.PComponents;
    pi(2,:) = foreGMFit.PComponents;


%     fgallclusters=assigncluster(params, im_data, squeeze(mu(2,:,:)), squeeze(sigma(2,:,:,:)), squeeze(pi(2,:)));
%     bgallclusters=assigncluster(params, im_data, squeeze(mu(1,:,:)), squeeze(sigma(1,:,:,:)), squeeze(pi(1,:)));
%     [~,fgallcluster] = max(fgallclusters,[],2);
%     [~,bgallcluster] = max(bgallclusters,[],2);
    bgallcluster = cluster(backGMFit, im_data);
    fgallcluster = cluster(foreGMFit, im_data);

    fprintf('\n\n\n\n');
    alpha = updateAlphaChoices(params, im_data, mu, sigma,pi, fgallcluster, bgallcluster, smoothIndices, smoothWeights);
end
toc

%drawClusters(fg,fgcluster);
im_data(logical(alpha==1),:) = 0;
im_data = reshape(im_data, [params.height params.width params.numColors]);
imshow(im_data);


