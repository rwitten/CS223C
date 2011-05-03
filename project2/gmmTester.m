function [] = gmmTester( )
    gmmOptions = statset(@gmdistribution);
    gmmOptions.MaxIter = 1000;
    K= 5; %number of mixtures
    n = 2; %dimensionality
    maxsamples = 100;

    

    samples = [];

    for i=1:K
        numberofsamples = floor(rand() * maxsamples);
        s = randn(n);
        sigma = .001 * (s' * s);
        samples = [samples; mvnrnd(rand(n,1), sigma, numberofsamples )];
    end

    %GMfit = gmdistribution.fit(samples, K, 'Options', gmmOptions);
    %idx = cluster(GMfit,samples);
    
    mu = rand(K,n);
    sigma = squeeze(makePositiveSemiD(1,K, n));
    pi = rand(K,1);
    
    params.K=K;
    params.numColors = n;
    for i=1:10000,
        sampleclusters = assignCluster(params, samples, mu, sigma,pi);
        [mu,sigma,pi]=updateGaussian(params, sampleclusters, samples);
        
        %waitforbuttonpress
    end
    clf('reset')
	drawClusters(samples, sampleclusters);
    
    
end