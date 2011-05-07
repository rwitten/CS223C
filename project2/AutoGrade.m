%Load image names
load('imgNames.mat', 'imageNames');

%Grab cut paramters
params.K = 5;
params.foreK = 5;
params.backK = 5;
params.numDirections = 8;
params.gamma = 50;
params.betaColCoeff = 2;
params.superEdgeSharpness = 10;
params.TotalIters = 20;
params.MaxIter = 1;
params.initIter = 1;
params.sharpAlpha = 0.2;
params.beInteractive = false;
params.useGMTools = false;
params.superSharpEdges = true;
params.useGT = true;

totalScore = 0;
for i = 1:length(imageNames)
    fprintf(imageNames{i});
    fprintf('\n');
    curScore = grabcut(imageNames{i}, params);
    totalScore = totalScore + curScore;
end

averageScore = totalScore/length(imageNames)