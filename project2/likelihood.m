function [l] = likelihood(x, mu, sigma, coeff)
    xoffset = bsxfun(@minus, x, mu');
    innerweirdness = sum(( xoffset*inv(sigma)) .*(xoffset),2);
    
    
    l = coeff*((2*pi)^(-length(mu)/2))*(abs(det(sigma))^(-.5)) ...
        *exp(-.5 *innerweirdness);
 
    %l =  1./sum((abs(x - repmat(mu',size(x,1),1))),2);

end
