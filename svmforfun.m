X1 = mvnrnd([1, 1] , eye(2),25);
X2 = mvnrnd([-1, -1] , eye(2),25);

jointdata = [X1; X2];
jointanswers = [ones(25,1); -ones(25,1)];

svmStruct = svmtrain(jointdata,jointanswers,'showplot',true);

SVMCLASSIFY(svmStruct, [1 1]);
SVMCLASSIFY(svmStruct, [-1 -1]);