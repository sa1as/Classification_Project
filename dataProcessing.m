%% cut data according to features and generate normalOperationData 
% normalOperationData is a structure of size equal to the normal operation phases 
% and contains the time series during that phase
% requires "add to path" of 'AllDataStruct' destination folder and returns
% SignalOperationData, changingPoint

load('AllDataStruct');

for j=1:length(dataSignals) - 1 % for every feature 
    
    colCounter = 1; 
    rowCounter = 1;

    for i = 1:(length(dataSignals{1}) - 1) % check each whole feature signal 

   % store signals from normal operation (label == -1)  
        if ( dataSignals{30}(i) == -1 && (dataSignals{30}(i + 1) - dataSignals{30}(i)) == 0 ) % if current and previous label == -1
            signalOperationData{colCounter}(rowCounter,j) = dataSignals{j}(i); % get all signals
            rowCounter = rowCounter + 1;
        elseif (dataSignals{30}(i) == -1 && (dataSignals{30}(i + 1) - dataSignals{30}(i)) ~= 0 ) % if label is not -1 change signal to be saved
            changingPoint(colCounter) = i;
            colCounter = colCounter + 1;
            rowCounter = 1;
        end

    end
    
end

clear colCounter rowCounter i counter j normalOperationData;

%% Split dat into two classes according to the prediction horizonhe
% The two classes are failure and noFailure

rollingSplit = 1;
horizon  =  5; % in minutes
signalWindow = horizon*60/5;
noFailureCounter = 1;
failureCounter = 1;
counter = 1;
signalNumber = 16;
clear failure;
clear noFailure;

if rollingSplit == 1
    disp('in the rolling process')
    
    for i = 1:length(signalOperationData)
        
        if length(signalOperationData{i}) < signalWindow % if signal is too short 
            failure(:,failureCounter) = zeros(signalWindow,1);
            failureCounter = failureCounter + 1;
        else
            
            failure(:,failureCounter) = signalOperationData{i}(end - signalWindow + 1:end, signalNumber); % save last "signalWindow length" data form vector as failed
            failureCounter = failureCounter +1;
            

            
            if length(signalOperationData{i}) >=  2*signalWindow % as long as the sliding window is bigger than the window
                for k = 1:signalWindow % 
                    failure(:,failureCounter) = signalOperationData{i}(end - signalWindow + 1 - k : end - k, signalNumber); % save last "signalWindow length" data form vector as failed
                    failureCounter = failureCounter +1;
                end
            else
                for k = 1:length(signalOperationData{i}) - signalWindow % 
                    failure(:,failureCounter) = signalOperationData{i}(end - signalWindow + 1 - k : end - k, signalNumber); % save last "signalWindow length" data form vector as failed
                    failureCounter = failureCounter +1;
                end
            end
            
            
            tempSignal = signalOperationData{i}(1:end - 2*k, signalNumber); % cut the signal used for the failure class - size of k (varying from 1 to signalWindow)
            leftSignalSize = floor(length(tempSignal)/2*signalWindow); % check if there is any signal left
            
            if leftSignalSize ~= 0
                
                for j=1:length(tempSignal) - signalWindow
                    noFailure{noFailureCounter} = signalOperationData{i}(j:j + signalWindow - 1, signalNumber); % get signals based on a sliding windowor
                    noFailureCounter = noFailureCounter + 1;
                end
                                
            else
                % noFailure{noFailureCounter} = [];
                noFailureCounter = noFailureCounter - 1;
            end
            
        end
        
    end

else % split non overlaping windows
    disp('in the window process')
    
    for i = 1: length(signalOperationData)
        
        if length(signalOperationData{i}) < signalWindow
            failure(:,i) = zeros(signalWindow,1);
        else
            failure(:,i) = signalOperationData{i}(end - signalWindow + 1:end, signalNumber); %save last "signalWindow length" data form vector as failed
            tempSignal = signalOperationData{i}(1:end - signalWindow, signalNumber);
            leftSignalSize = floor(length(tempSignal)/signalWindow);
            
            if leftSignalSize ~= 0
                for j=1:leftSignalSize
                    noFailure{counter} = signalOperationData{i}((j-1)*signalWindow + 1: j*signalWindow, signalNumber); % get signals for every window
                    counter = counter + 1;
                end
                noFailure{counter} = signalOperationData{i}(leftSignalSize*signalWindow:end - signalWindow, signalNumber); %save a signal window of the leftover
                counter = counter + 1;
            else
                noFailure{counter} = [];
                counter = counter + 1;
            end
        end
    end
end
    
clear i j k horizon leftSignalSize signalWindow tempSignal noFailureCounter failureCounter rollingSplit signalNumber;



%% noFailure without zeros

counter = 1;
for i = 1:length(noFailure)
    if length(noFailure{i}) == 60
        updatedNoFailure(:,counter) = noFailure{i};
        counter = counter + 1;
    end
end


feature_array_noFailure = featExtract(updatedNoFailure);
feature_array_failure = featExtract(failure);

%% train classifier 

% don't forget to normalize data although there are of equal length and
% might not be necessary 

%% -------- Train SVM choosing a Kernel ----------  

if RUNSVM == 1 
    %[1, 5, 9, 10, 13, 23];
    %FeatureVector = [5, 9, 10, 13, 23];
    TrainFeat = [feature_array_failure(1:350,:); feature_array_noFailure(301:650,:)];
    TrainingLabels = [ones(350,1); zeros(350,1)];
    TestingLabels = [ones(150,1); zeros(150,1)];
    TestingFeatures = [feature_array_failure(351:500,:); feature_array_noFailure(501:650,:)];
    
    %TrainFeat = TrainingFeatures(:,FeatureVector);
    
    SVM = fitcsvm(TrainFeat, TrainingLabels, 'BoxConstraint', 1, 'KernelFunction','rbf', 'KernelScale', 'auto');
    [PredicitionLabels, PredictionScores] = predict(SVM, TestingFeatures);

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


