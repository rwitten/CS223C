function [ fgcluster,fg,bgcluster,bg ] = updateClusterChoices(params,alpha, im_data, mu, sigma)
    alphavec = reshape(alpha,params.numPixels,1);
    im_vec = reshape(im_data, params.numPixels,params.numColors);
    
    bg=im_vec(logical(alphavec==1),:);
    fg = im_vec(logical(alphavec==2),:);
    
    fgclusters=assignCluster(params, fg, mu(2,:,:),sigma(2,:,:,:));
    bgclusters=assignCluster(params, bg, mu(1,:,:),sigma(2,:,:,:));
    [~,fgcluster] = max(fgclusters,[],2);
    [~,bgcluster] = max(bgclusters,[],2);
    
%     likelihoods = zeros(alphavec.K, 2, size(im_data,1) * size(im_data,2)
%     for k=1:params.K,
%         for a = 1:2,
%             
%     
%     mu = squeeze(mu(1,1,:));
%     sigma = squeeze(sigma(1,1,:,:));
%     sigma = eye(3,3);
%     
%     x = ones(size(im_data,1) * size(im_data,2),3);
%     size(((x*sigma).*x)')
%     sum( ((x*sigma).*x)' );
%     k=1;
    
end

function [l] = likelihood(x, mu, sigma)
    
    xoffset = x;
    for i=1:size(x,3) %number of channels
        xoffset(:,:,i) = xoffset(:,:,i) - mu(i);
    end
    
    
    innerweirdness = sum(( xoffset*inv(sigma)) .*(xoffset),2);
    
    l = (2*pi)^(-length(mu)/2)*abs(det(sigma))^(-.5) ...
        *exp(-.5 *innerweirdness);
end

function [likelihoods] = assignCluster(params,vector,mu,sigma)
    mu = squeeze(mu);
    sigma = squeeze(sigma);
    
    likelihoods = zeros(size(vector,1), params.K);
    for i = 1:params.K,
        likelihoods(:,i)=likelihood(vector, squeeze(mu(i,:))', squeeze(sigma(i,:,:)));
    end
end

