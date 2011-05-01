function [] = drawClusters( pixels, clusters )
%pixels is a n by numColors matrix of pixels, a clusters a n by 1 set of
%assignments.

pixels = pixels(:,1:2); % 2D display

perm = 1:size(pixels,1);
pixels = pixels(perm,:);
clusterdrawn = clusters(perm);

elements = pixels(logical(clusterdrawn==1),:);
scatter(elements(:,1),elements(:,2),'b');
hold on
elements = pixels(logical(clusterdrawn==2),:);
scatter(elements(:,1),elements(:,2),'r');
hold on
elements = pixels(logical(clusterdrawn==3),:);
scatter(elements(:,1),elements(:,2),'g');
hold on
elements = pixels(logical(clusterdrawn==4),:);
scatter(elements(:,1),elements(:,2),'p');
hold on
elements = pixels(logical(clusterdrawn==5),:);
scatter(elements(:,1),elements(:,2),'y');

end

