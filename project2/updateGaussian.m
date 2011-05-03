function [mu, sigma, pi] = updateGaussian(params, clusters, pixels)
    mu = zeros(params.K, params.numColors);
    sigma = zeros(params.K, params.numColors, params.numColors);
    pi = zeros(params.K,1);
    
    for k = 1:params.K,
        cluster_pixels = pixels(logical(clusters==k),:);
        
        if sum(clusters==k)>=1
            pi(k) = size(cluster_pixels,1) / size(pixels,1);
            mu(k,:) = sum(cluster_pixels,1)/size(cluster_pixels,1);


            unbiased_pixels = bsxfun(@minus, cluster_pixels, mu(k,:));

            ninv = 1/(size(unbiased_pixels,1)-1);
            sigma(k,:,:)= eye(params.numColors)*1e-2 +ninv*unbiased_pixels' *  unbiased_pixels;
            %sigma(k,:,:) = eye(params.numColors);
        else
            pi(k) = 1/params.K;
            mu(k,:) = pixels(floor(rand()*size(pixels,1))+1, :);
            sigma(k,:,:) = 1e-2 * eye(params.numColors);

        end
    end
end