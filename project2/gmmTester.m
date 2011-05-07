function [] = gmmTester( )
    gmmOptions = statset(@gmdistribution);
    gmmOptions.MaxIter = 1000;
    K= 2; %number of mixtures
    n = 2; %dimensionality
    maxsamples = 100;

    

    samples = [];

    for i=1:K
        numberofsamples = floor(rand() * maxsamples);
        numberofsamples = maxsamples;
        s = randn(n);
        sigma = 1e-3 * (s' * s);
        samples = [samples; mvnrnd(rand(n,1), sigma, numberofsamples )];
    end

    %GMfit = gmdistribution.fit(samples, K, 'Options', gmmOptions);
    %idx = cluster(GMfit,samples);
    
    mu = rand(K,n);
    sigma = squeeze(makePositiveSemiD(1,K, n));
    pi = ones(K,1)/K;
    
    params.K=K;
    params.numColors = n;
    for i=1:100,
        disp 'iteration!'
        sampleclusters = assignCluster(params.K, samples, mu, sigma,ones(K,1)/K);
        [mu,sigma,pi]=updateGaussian(params, params.K,sampleclusters, samples);
%          mu(1,:)
%          squeeze(sigma(1,:,:))
%          mu(2,:)
%          squeeze(sigma(2,:,:))
%          clf('reset')
%      	drawClusters(samples, sampleclusters);
%          waitforbuttonpress
    end
    clf('reset')
	drawClusters(samples, sampleclusters);
    
    
end
