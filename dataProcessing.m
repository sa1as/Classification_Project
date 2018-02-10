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


feature_array_noFailure = featExtract(updatedNoFailure); % extract features
feature_array_failure = featExtract(failure);

%% train classifier 

% don't forget to normalize data although there are of equal length and
% might not be necessary 

classifier = 1; % choose classifier "1" for SVM

FeatureVector = [1:24]; % choose feature vector for training out of 24 pre-calculated features
trainingVectorSize = 10000; % define size of training vector
testingVectorSize = 100;

randomSample = randsample(length(feature_array_failure), trainingVectorSize); % randomly choose class one (failure) samples for training
trainingFailureFeatureVector = feature_array_failure(randomSample, FeatureVector);
randomSample2 = randsample( setdiff(1:length(feature_array_failure), randomSample), testingVectorSize)'; % randomly choose class one (failure) samples for training
testingFailureFeatureVector = feature_array_failure(randomSample2,FeatureVector); 

randomSample = randsample(length(feature_array_failure), trainingVectorSize); % randomly choose class two (noFailure) sample for training
trainingNoFailureFeatureVector = feature_array_noFailure(randomSample, FeatureVector);
randomSample2 = randsample( setdiff(1:length(feature_array_noFailure), randomSample), testingVectorSize)'; % randomly choose class one (failure) samples for training
testingNoFailureFeatureVector = feature_array_noFailure(randomSample2,FeatureVector); 

trainingLabels = [zeros(trainingVectorSize,1); ones(trainingVectorSize,1)];
testingLabels = [zeros(testingVectorSize,1); ones(testingVectorSize,1)];
%testingLabels = [zeros(trainingVectorSize,1); ones(trainingVectorSize,1)];
trainFeat = [trainingFailureFeatureVector; trainingNoFailureFeatureVector]; %final training feature vector combined 
testingFeat = [testingFailureFeatureVector; testingNoFailureFeatureVector]; %final training feature vector combined 
%testingFeat = [trainingFailureFeatureVector; trainingNoFailureFeatureVector]; %final training feature vector combined 


learnAndTest(trainFeat, trainingLabels,testingFeat ,testingLabels, classifier)

