function [ mu,sigma,pi] = updateClusterParameters(params, fgcluster,fg,...
    bgcluster,bg)
    [mu_fg, sigma_fg, pi_fg] = updateGaussian(params, fgcluster, fg);
    [mu_bg, sigma_bg, pi_bg] = updateGaussian(params, bgcluster, bg);
    mu(1,:,:) = mu_bg;
    mu(2,:,:) = mu_fg;
    sigma(1,:,:,:) = sigma_bg;
    sigma(2,:,:,:) = sigma_fg;
    pi(1,:) = pi_fg;
    pi(2,:) = pi_bg;
end
