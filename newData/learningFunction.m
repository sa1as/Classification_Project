%{

*******************************************************************
*                                                                 *
*                data preparation and learning                    *
*                                                                 *
*******************************************************************
%}

clear all;
clc;

addpath(genpath('/home/stefanos/Documents/MatlabCode/'));
addpath(genpath('/home/stefanos/Documents/MatlabData/'));


% ---------- Load Training data 
load('Plug_Signal_Struct_60s_uncut_Human.mat');
PlugTraining = Signal_Struct; 

load('ABB_Signal_Struct_60s_uncut_Human.mat');
ABBTraining = Signal_Struct;

% ---------- Load Training labels
load('Labels_Plug_60s_Human.mat');
PlugLabels = labels;

load('Labels_ABB_60s_Human.mat');
ABBLabels = labels;

% ---------- Load Testing data
load('Plug_Signal_Struct_50s_Robot.mat');
PlugTesting = Signal_Struct;

load('ABB_Signal_Struct_53s_final_v2_Robot.mat');
ABBTesting = Signal_Struct;

% ---------- Load Testing labels 
load('Labels_Plug_50s_Robot.mat');
PlugTestingLabels = labels;

load('Labels_ABB_53s_v2_Robot.mat');
ABBTestingLabels = labels;



TrainingFeatures = featExtract(ABBTraining); % PlugTraining, ABBTraining

TrainingMean = mean(TrainingFeatures, 1);
TrainingSigma = std(TrainingFeatures, 1);
ScaledTrainingFeatures = (TrainingFeatures - TrainingMean)./TrainingSigma;
TestingFeatures = featExtract(ABBTesting);  % PlugTesting, ABBTesting
ScaledTestingFeatures = (TestingFeatures - TrainingMean)./TrainingSigma;
TrainingFeatures = ScaledTrainingFeatures;
TestingFeatures = ScaledTestingFeatures;

TrainingLabels = ABBLabels; %  PlugLabels, ABBLabels
TestingLabels = ABBTestingLabels; % PlugTestingLabels, ABBTestingLabels


RUNSVM = 0;  % run SVM classifier
RunNN = 0;   % run NN classifier
plotVar = 0; % plot SVM


%% -------- Train SVM choosing a Kernel ----------  

if RUNSVM == 1 
    %[1, 5, 9, 10, 13, 23];
    FeatureVector = [5, 9, 10, 13, 23];
    TrainFeat = TrainingFeatures(:,FeatureVector);
    SVM = fitcsvm(TrainFeat, TrainingLabels, 'BoxConstraint', 1, 'KernelFunction','rbf', 'KernelScale', 'auto');
    [PredicitionLabels, PredictionScores] = predict(SVM, TestingFeatures(:,FeatureVector));

    SV = SVM.SupportVectors;
    SVM.SupportVectorLabels;

    if plotVar == 1
    %%{
    % plot training data points
    gscatter(TrainingFeatures(:,FeatureVector(1)), TrainingFeatures(:,FeatureVector(2)), TrainingLabels, 'rb');  % plot training data by group -> gives different colours
    hold on
    plot(SV(:,1),SV(:,2),'ko','MarkerSize',10);

    % plot SVM hyperplane
        d = 0.1;
        [x1Grid,x2Grid] = meshgrid(-5:d:5, -5:d:5);
        xGrid = [x1Grid(:),x2Grid(:)];
        [~,scores] = predict(SVM, xGrid);
        contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
        gscatter(TestingFeatures(:,FeatureVector(1)), TestingFeatures(:,FeatureVector(2)), TestingLabels, 'gk'); % plot testing data
        %}
    end
    
    Predictions = ~xor(TestingLabels, PredicitionLabels); % correct predictions vec
    PredNum = sum(Predictions); % correct prediction num 
    Accuracy = sum(PredNum/size(TestingLabels, 1))
    
    misslabeled = find(Predictions == 0);
    FalseNegatives = size(find(PredicitionLabels(misslabeled) == 0), 1)
    FalsePositives = size(find(PredicitionLabels(misslabeled) == 1), 1)
    figure, plotconfusion(TestingLabels',PredicitionLabels')


end

%% -------- Train Classification NN ----------  
if RunNN == 1
   
    % 'trainlm' is usually fastest.
    % 'trainbr' takes longer but may be better for challenging problems.
    % 'trainscg' uses less memory. Suitable in low memory situations.
    
    trainFcn = 'trainscg';  % choose training function - Scaled conjugate gradient backpropagation
    feat = [5, 9, 10, 13, 23];
    TrainingFeatures = TrainingFeatures(:,feat)';  % training 
    TestingFeatures = TestingFeatures(:,feat)';
    %labels = Traininglabels';  %  
    
    hiddenLayerSize = 10;
    net = patternnet(hiddenLayerSize, trainFcn); % define network structure  --> classification network
%   net = fitnet(hiddenLayerSize,trainFcn); % fitting network

    
    % Setup Division of Data for Training, Validation, Testing
    % For a list of all data division functions type: help nndivide
    net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideMode = 'sample';  % Divide up every sample
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    
    % train network
    [net,tr] = train(net,TrainingFeatures,TrainingLabels');  
    performance = perform(net, TestingFeatures, TestingLabels');
    
    % test the network
    ytraining = net(TrainingFeatures);
    ytesting = sim(net,TestingFeatures);
    

    
    % Plots - % Uncomment these lines to enable various plots.
    %figure, plotperform(tr)
    %figure, plottrainstate(tr)
    %figure, ploterrhist(e)
    figure, plotconfusion(TrainingLabels', ytraining)
    figure, plotconfusion(TestingLabels',ytesting)
    %figure, plotroc(t,y)
    
end

