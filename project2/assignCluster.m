function [cluster] = assignCluster(K,vector,mu,sigma, pi)
    mu = squeeze(mu);
    sigma = squeeze(sigma);
    pi  = squeeze(pi);
    
    likelihoods = zeros(size(vector,1), K);
    for i = 1:K,
        likelihoods(:,i)=likelihood(vector, squeeze(mu(i,:))', squeeze(sigma(i,:,:)),pi(i));
    end
    [~, cluster] = max(likelihoods,[],2);
end