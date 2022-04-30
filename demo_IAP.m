
clear;clc

load IAPRTC-12.mat;
XTrain = I_tr;  YTrain = T_tr;
XTest  = I_te;  YTest  = T_te;
LTrain = L_tr;  LTest  = L_te;

load noisyLabel0.4.mat
ratio = 0.4;
LTrain_n = LTrain_n; % noisy label matrix

XTest  = bsxfun(@minus, XTest, mean(XTrain, 1)); 
XTrain = bsxfun(@minus, XTrain, mean(XTrain,1));
YTest  = bsxfun(@minus, YTest, mean(YTrain, 1));
YTrain = bsxfun(@minus, YTrain, mean(YTrain,1));

[XKTrain,XKTest] = Kernelize(XTrain, XTest, 1000) ; [YKTrain,YKTest]=Kernelize(YTrain, YTest, 1000);
XKTest = bsxfun(@minus, XKTest, mean(XKTrain, 1)); XKTrain = bsxfun(@minus, XKTrain, mean(XKTrain, 1));
YKTest = bsxfun(@minus, YKTest, mean(YKTrain, 1)); YKTrain = bsxfun(@minus, YKTrain, mean(YKTrain, 1));

%%
param.beta  = 5e-3;
param.gamma = 1e-2;
param.mu = 1e-1;
param.phi  = 1e-3;
param.lambda = 1e-4;
param.eta0 = .9; param.eta1 = .05; param.eta2 = .05;

param.iter = 8;
nbitset  = [16,32,64,128];

eva_info = cell(1,length(nbitset));
for bit = 1:length(nbitset) 
    
   param.nbits = nbitset(bit);
   [B, B1, B2] = WASH(XKTrain, YKTrain, LTrain_n, param, XKTest, YKTest);

   DHamm = hammingDist(B1, B);
[~, orderH] = sort(DHamm, 2);
 eva_info_.Image_to_Text_MAP = mAP(orderH', LTrain, LTest);
% [eva_info_.Image_to_Text_precision, eva_info_.Image_to_Text_recall] = precision_recall(orderH', LTrain, LTest);
% eva_info_.Image_To_Text_Precision_K = precision_at_k(orderH',  LTrain, LTest,1000);

 
DHamm = hammingDist(B2, B);
[~, orderH] = sort(DHamm, 2);
eva_info_.Text_to_Image_MAP = mAP(orderH', LTrain, LTest);
% [eva_info_.Text_to_Image_precision, eva_info_.Text_to_Image_recall] = precision_recall(orderH', LTrain, LTest);
% eva_info_.Text_To_Image_Precision_K = precision_at_k(orderH',  LTrain, LTest, 1000);

eva_info{1,bit} = eva_info_;
Image_to_Text_MAP = eva_info_.Image_to_Text_MAP;
Text_to_Image_MAP = eva_info_.Text_to_Image_MAP;  
fprintf('WASH %.1f RON %d bits -- Image_to_Text_MAP: %.4f ; Text_to_Image_MAP: %.4f ; \n',ratio,param.nbits,Image_to_Text_MAP,Text_to_Image_MAP);
end
