function [KTrain, KTest] = Kernelize(Train,Test,n_anchor)
    [n,~]=size(Train);
    [nT,~]=size(Test);
    rand('seed',2021);
    anchor = Train(randsample(n,n_anchor),:);
   
    KTrain = sqdist(Train',anchor');
    sigma = mean(mean(KTrain,2));
    KTrain = exp(-KTrain/(2*sigma));  
    mvec = mean(KTrain);
    KTrain = KTrain-repmat(mvec,n,1);
    
    KTest = sqdist(Test',anchor');
    KTest = exp(-KTest/(2*sigma));
    KTest = KTest-repmat(mvec,nT,1);
    
end