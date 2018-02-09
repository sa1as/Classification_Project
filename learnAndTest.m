%% learn an SVM model
function  learnAndTest(feature_array_failure, feature_array_noFailure, classifier)

%% -------- Train SVM choosing a Kernel ----------  
RUNSVM = classifier;

if RUNSVM == 1 
    %[1, 5, 9, 10, 13, 23];
    FeatureVector = [5, 9, 10, 13, 23];
    TrainFeat = [feature_array_failure(1:350,FeatureVector); feature_array_noFailure(301:650,FeatureVector)];
    TrainingLabels = [zeros(350,1); ones(350,1)];
    TestingLabels = [zeros(150,1); ones(150,1)];
    TestingFeatures = [feature_array_failure(351:500,FeatureVector); feature_array_noFailure(501:650,FeatureVector)];
    
    %TrainFeat = TrainingFeatures(:,FeatureVector);
    
    SVM = fitcsvm(TrainFeat, TrainingLabels, 'BoxConstraint', 1, 'KernelFunction','rbf', 'KernelScale', 'auto');
    [PredicitionLabels, PredictionScores] = predict(SVM, TestingFeatures);

    SV = SVM.SupportVectors;
    SVM.SupportVectorLabels;

    if  0
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




