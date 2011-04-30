function [ mu,sigma,pi] = updateClusterParameters(params, alpha, im_data,fgcluster,fg,...
    bgcluster,bg)
    [mu_fg, sigma_fg, pi_fg] = updateGaussian(params, fgcluster, fg);
    [mu_bg, sigma_bg, pi_bg] = updateGaussian(params, fgcluster, fg);
    mu(1,:,:) = mu_bg;
    mu(2,:,:) = mu_fg;
    sigma(1,:,:,:) = sigma_bg;
    sigma(2,:,:,:) = sigma_fg;
    pi(1,:) = pi_fg;
    pi(2,:) = pi_bg;
end

function [mu, sigma, pi] = updateGaussian(params, clusters, pixels)
    mu = zeros(params.K, params.numColors);
    sigma = zeros(params.K, params.numColors, params.numColors);
    pi = zeros(params.K,1);
    
    for k = 1:params.K,
        cluster_pixels = pixels(logical(clusters==k),:);
        
        if sum(vec(clusters)==k)>=1
            pi(k) = size(cluster_pixels,1) / size(pixels,1);
            mu(k,:) = sum(cluster_pixels,1)/size(pixels,1);
            
            unbiased_pixels = zeros(size(cluster_pixels,1),params.numColors);
            size(unbiased_pixels)
            for i = 1:params.numColors,
                unbiased_pixels(:,:,i) = cluster_pixels(:,:,i) + mu(k,i);
            end
            sigma(k,:,:)=unbiased_pixels' *  unbiased_pixels;
        else
            pi(k) = 0;
            mu(k,:) = .5*rand(1,params.numColors);
            sigma(k,:,:) = eye(params.numColors, params.numColors);
        end
    end
end
