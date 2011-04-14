function [detector] = detectorTrain(gt, examples)

binarygt = 2*((gt>0)-.5);
detector = liblineartrain(binarygt', sparse(examples), '-s 2 -B 1');

[predicted_label_test, accuracy] ...
       = liblinearpredict(binarygt',sparse(examples),detector);

 scores =  [examples, ones(size(examples,1),1)] * detector.w';
   
 if sum(abs(2*((scores > 0 )-.5) - predicted_label_test)) < 1,
     detector.multiplier = 1;
 else
     detector.multiplier = -1;
 end