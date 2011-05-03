function [mu, sigma, pi] = updateGaussian(params, clusters, pixels)
    mu = zeros(params.K, params.numColors);
    sigma = zeros(params.K, params.numColors, params.numColors);
    pi = zeros(params.K,1);
    
    for k = 1:params.K,
        cluster_pixels = pixels(logical(clusters==k),:);
        
        if sum(clusters==k)>=1
            pi(k) = size(cluster_pixels,1) / size(pixels,1);
            mu(k,:) = sum(cluster_pixels,1)/size(cluster_pixels,1);


            unbiased_pixels = cluster_pixels - repmat(mu(k,:), size(cluster_pixels,1),1);

            sigma(k,:,:)= eye(params.numColors)*1e-5 +unbiased_pixels' *  unbiased_pixels;
            %sigma(k,:,:) = eye(params.numColors);
        else
            pi(k) = 0;
            mu(k,:) = rand(1,params.numColors);
            sigma(k,:,:) = eye(params.numColors, params.numColors);
        end
    end
end