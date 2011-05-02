function [l] = likelihood(x, mu, sigma)
    
    xoffset = x;
    for i=1:size(x,3) %number of channels
        xoffset(:,i) = xoffset(:,i) - mu(i);
    end
    
    innerweirdness = sum(( xoffset*inv(sigma)) .*(xoffset),2);
    
    l = (2*pi)^(-length(mu)/2)*abs(det(sigma))^(-.5) ...
        *exp(-.5 *innerweirdness);
end