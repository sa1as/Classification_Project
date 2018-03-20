%% learn an SVM model
function  [Accuracy, FalseNegatives, FalsePositives] = learnAndTest(trainFeat, trainingLabels,testingFeat ,testingLabels, FeatureVector, classifier)

%% -------- Train SVM choosing a Kernel ----------  
RUNSVM = classifier;

if RUNSVM == 1 
    %[1, 5, 9, 10, 13, 23];
   % FeatureVector = [5, 9, 10, 13, 23];
    %TrainFeat = [feature_array_failure(1:350,FeatureVector); feature_array_noFailure(301:650,FeatureVector)];
    %TrainingLabels = [zeros(350,1); ones(350,1)];
    %TestingLabels = [zeros(150,1); ones(150,1)];
    %TestingFeatures = [feature_array_failure(351:500,FeatureVector); feature_array_noFailure(501:650,FeatureVector)];
    
    %TrainFeat = TrainingFeatures(:,FeatureVector);
    
    SVM = fitcsvm(trainFeat, trainingLabels, 'BoxConstraint', 1, 'KernelFunction','rbf', 'KernelScale', 'auto');
    [PredicitionLabels, PredictionScores] = predict(SVM, testingFeat);

    SV = SVM.SupportVectors;
    SVM.SupportVectorLabels;

    if  0
    %%{
    % plot training data points
    gscatter(trainFeat(:,1), trainFeat(:,2), trainingLabels, 'rb');  % plot training data by group -> gives different colours
    hold on
    plot(SV(:,1),SV(:,2),'ko','MarkerSize',10);

    % plot SVM hyperplane
        d = 0.1;
        [x1Grid,x2Grid] = meshgrid(-5:d:5, -5:d:5);
        xGrid = [x1Grid(:),x2Grid(:)];
        [~,scores] = predict(SVM, xGrid);
        contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
        gscatter(testingFeat(:,1), testingFeat(:,2), testingLabels, 'gk'); % plot testing data
        %}
    end
    
    Predictions = ~xor(testingLabels, PredicitionLabels); % correct predictions vec -> 1 is correclty predicted 
    PredNum = sum(Predictions); % correct prediction num 
    % positive class "1" noFailure 
    misslabeled = find(Predictions == 0); % find wrong predictions
    FalseNegatives = size(find(PredicitionLabels(misslabeled) == 0), 1); % from the misslabeled find those predicted 0 -> FN signals that should be possitive label 1
    
    FN = misslabeled(find(PredicitionLabels(misslabeled) == 0)); % signals miss labeled 1 -> nofailure  

    FalsePositives = size(find(PredicitionLabels(misslabeled) == 1), 1);
    
    
    %failureSignalSizes = diff(failureSignalIndex5min);
    FP = misslabeled(find(PredicitionLabels(misslabeled) == 1)); 

    failureFullTestingSignals = [];
    for ii = 1:size(failureTestingRandomSample)
        failureFullTestingSignals = [failureFullTestingSignals ones(1,failureSignalSizes(failureTestingRandomSample(ii)))*ii];
    end
    [fa,b]=hist(failureFullTestingSignals(FP),unique(failureFullTestingSignals(FP)))
    noFailureSignalSizes = diff(noFailureSignalIndex5min);
    noFailureFullTestingSignals = [];
    for ii = 1:size(noFailureTestingRandomSample)
        noFailureFullTestingSignals = [noFailureFullTestingSignals ones(1,noFailureSignalSizes(noFailureTestingRandomSample(ii)+1))*ii];
    end
    [noa,b]=hist(noFailureFullTestingSignals(FN),unique(noFailureFullTestingSignals(FN)))

    
    negativeClass = 1 - FalseNegatives/sum(testingLabels, 1)
    positiveClass = 1 - FalsePositives/(size(testingLabels, 1) - sum(testingLabels, 1))
    Accuracy = (1 - FalseNegatives/sum(testingLabels, 1) + 1 - FalsePositives/(size(testingLabels, 1) - sum(testingLabels, 1)))/2

    %FPSignals = failureSignalNumber(FP);
    %FNSignals = noFailureSignalNumber(FN);
    
    %FNNum = size(unique(FNSignals.','rows').'); % remove dublicates
    %FPNum = size(unique(FPSignals.','rows').'); % remove dublicates

    
    figure, plotconfusion(testingLabels',PredicitionLabels')
    
    
        % 'trainlm' is usually fastest.
    % 'trainbr' takes longer but may be better for challenging problems.
    % 'trainscg' uses less memory. Suitable in low memory situations.
    if 0
        trainFcn = 'trainscg';  % choose training function - Scaled conjugate gradient backpropagation
        %feat = [5, 9, 10, 13, 23];
        TrainingFeatures = trainFeat(:,feat)';  % training
        TestingFeatures = testingFeat(:,feat)';
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
        [net,tr] = train(net,trainFeat,trainingLabels');
        performance = perform(net, testingFeat, testingLabels');
        
        % test the network
        ytraining = net(trainFeat);
        ytesting = sim(net,testingFeat);
        

    end


end




