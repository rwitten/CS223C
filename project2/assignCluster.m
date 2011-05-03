function [cluster] = assignCluster(params,vector,mu,sigma, pi)
    mu = squeeze(mu);
    sigma = squeeze(sigma);
    pi  = squeeze(pi);
    
    likelihoods = zeros(size(vector,1), params.K);
    for i = 1:params.K,
        likelihoods(:,i)=likelihood(vector, squeeze(mu(i,:))', squeeze(sigma(i,:,:)),1);
    end
    [~, cluster] = max(likelihoods,[],2);
end