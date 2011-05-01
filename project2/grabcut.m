function grabcut(im_name)
im_name='banana1.bmp';




% % convert the pixel values to [0,1] for each R G B channel.
im_data = double(imread(im_name)) / 255;

%im_data = im_data(1:100,1:100,:);
% % display the image
% imagesc(im_data);
% 
% % a bounding box initialization
% disp('Draw a bounding box to specify the rough location of the foreground');
% set(gca,'Units','pixels');
% ginput(1);
% p1=get(gca,'CurrentPoint');fr=rbbox;p2=get(gca,'CurrentPoint');
% p=round([p1;p2]);
% xmin=min(p(:,1));xmax=max(p(:,1));
% ymin=min(p(:,2));ymax=max(p(:,2));
 [im_height, im_width, channel_num] = size(im_data);

% xmin = max(xmin, 1);
% xmax = min(im_width, xmax);
% ymin = max(ymin, 1);
% ymax = min(im_height, ymax);

xmin = 32;
xmax = 606;
ymin = 29;
ymax = 421;

bbox = [xmin ymin xmax ymax];
%line(bbox([1 3 3 1 1]),bbox([2 2 4 4 2]),'Color',[1 0 0],'LineWidth',1);

if channel_num ~= 3
    disp('This image does not have all the RGB channels, you do not need to work on it.');
    return;
end

params.K = 5;
params.numColors = channel_num;
params.numPixels= im_height * im_width;


trimap = zeros(im_height,im_width);

for h = 1 : im_height
     for w = 1 : im_width
         if (w > xmin) && (w < xmax) && (h > ymin) && (h < ymax)
	     trimap(h,w) = 3; %this means that its T_U or the initial foreground
             alpha(h,w) = 2; %2 means that its T_U or the initial foreground
         else
             alpha(h,w) = 1; %1 means its in T_B or the initial background
             trimap(h,w) = 1; %this means its in T_B or the initial background
         end
     end
end

%alpha = repmat(trimap==3, numel(trimap));

mu = rand(2,params.K,params.numColors);
sigma = makePositiveSemiD(2,params.K, params.numColors);
pi = zeros(2, params.K);

% grabcut algorithm
fprintf('*************************\n');
fprintf('****grabcut algorithm****\n');
fprintf('*************************\n\n\n\n');

for iter=1:100%bs stopping criteria
    fprintf('we are on iteration %d\n', iter);
    
    fprintf('we are updating the cluster choices\n');
    [ fgcluster,fg,bgcluster,bg ] = updateClusterChoices(params,alpha, im_data,...
        mu, sigma);
    
    fprintf('we are updating the cluster parameters\n');
    [mu, sigma,pi] = updateClusterParameters(params, im_data,fgcluster,fg,bgcluster,bg);
    
    
    fprintf('\n\n\n\n');
    %alpha = updateBackgroundForegroundChoices(params, alpha,im_data, mu, sigma,pi,xmin, xmax, ymin, ymax);
end

%drawClusters(fg,fgcluster);


