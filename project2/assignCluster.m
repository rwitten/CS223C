function [likelihoods] = assignCluster(params,vector,mu,sigma)
    mu = squeeze(mu);
    sigma = squeeze(sigma);
    
    likelihoods = zeros(size(vector,1), params.K);
    for i = 1:params.K,
        likelihoods(:,i)=likelihood(vector, squeeze(mu(i,:))', squeeze(sigma(i,:,:)));
    end
end