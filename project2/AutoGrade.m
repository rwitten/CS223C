%Load image names
load('imgNames.mat', 'imageNames');

%Grab cut paramters
params.K = 5;
params.foreK = 10;
params.backK = 10;
params.numDirections = 8;
params.gamma = 50;
params.betaColCoeff = 2;
params.superEdgeSharpness = 10;
params.TotalIters = 20;
params.MaxIter = 1;
params.initIter = 1;
params.sharpAlpha = 0.2;
params.beInteractive = false;
params.useGMTools = true;
params.superSharpEdges = false;
params.useGT = true;
params.innerIters = 1;
totalScore = 0;
totalHits = 0;
for i = 1:length(imageNames)
    fprintf(imageNames{i});
    fprintf('\n');
    try
        curScore = grabcut(imageNames{i}, params)
        totalScore = totalScore + curScore;
        totalHits = totalHits + 1;
    end
end

averageScore = totalScore/totalHits
