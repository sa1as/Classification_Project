%% cut data according to features and generate normalOperationData 
% normalOperationData is a structure of size equal to the normal operation phases 
% and contains the time series during that phase
% requires "add to path" of 'AllDataStruct' destination folder and returns
% SignalOperationData, changingPoint

load('AllDataStruct');


for j=1:length(dataSignals) - 1 % for every signal 
    
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

%% Split dat into two classes according to the prediction horizon
% The two classes are failure and noFailure

rollingSplit = 0;
horizon  =  5; % in minutes defines the window from which features are extracted
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
        
        failureSignalIndex(i) = failureCounter; % its the index where the source singal is different 
        noFailureSignalIndex(i) = noFailureCounter;
        
        if length(signalOperationData{i}) < signalWindow % if signal is too short 
            %failure(:,failureCounter) = zeros(signalWindow,1);
            %failureCounter = failureCounter + 1;
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
                    noFailure(:,noFailureCounter) = signalOperationData{i}(j:j + signalWindow - 1, signalNumber); % get signals based on a sliding window
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
failureSignalIndex = sort(failureSignalIndex);
noFailureSignalIndex = sort(noFailureSignalIndex);

failureSignalIndex = unique(failureSignalIndex.','rows').';
noFailureSignalIndex = unique(noFailureSignalIndex.','rows').'; % remove dublicates


%% noFailure without zeros

counter = 1;
for i = 1:length(noFailure)
    if length(noFailure{i}) == 60 % 60 for 5 minutes // 540 for 45 minutes
        updatedNoFailure(:,counter) = noFailure{i};
        counter = counter + 1;
    end
end


feature_array_noFailure10min = featExtractMod(noFailure10min); % extract features
feature_array_failure10min = featExtractMod(failure10min);

% standardise features


%% extract training and testing set / process features

trainingVectorSize = 10000; % define size of training vector
testingVectorSize = 1000;
failureSignalIndex = failureSignalIndex5min;
noFailureSignalIndex = noFailureSignalIndex5min;


% === choose training and testing set ===

feature_array_failure = feature_array_failure5min;
feature_array_noFailure = feature_array_noFailure5min;

feature_array_failure(isnan(feature_array_failure)) = 0; % remove 'NaN' and 'Inf' values should probably be removed
feature_array_noFailure(isnan(feature_array_noFailure)) = 0; 
feature_array_failure(isinf(feature_array_failure)) = 0;
feature_array_noFailure(isinf(feature_array_noFailure)) = 0;

% define training data
failureTrainingRandomSample = randsample(length(feature_array_failure), trainingVectorSize); % randomly choose class one (failure) samples for training
failureTrainingFeatureVector = feature_array_failure(failureTrainingRandomSample, :);
noFailureTrainingRandomSample = randsample(length(feature_array_noFailure), trainingVectorSize); % randomly choose class two (noFailure) sample for training
noFailureTrainingFeatureVector = feature_array_noFailure(noFailureTrainingRandomSample, :);
trainingLabels = [zeros(trainingVectorSize,1); ones(trainingVectorSize,1)]; % label = "0" -> failure / negative

% define testing data
failureTestingRandomSample = sort(randsample(setdiff(1:length(feature_array_failure), failureTrainingRandomSample), testingVectorSize)'); % randomly choose class one (failure) samples for training
failureTestingFeatureVector = feature_array_failure(failureTestingRandomSample, :);
noFailureTestingRandomSample = sort(randsample( setdiff(1:length(feature_array_noFailure), noFailureTrainingRandomSample), testingVectorSize)'); % randomly choose class one (failure) samples for training
noFailureTestingFeatureVector = feature_array_noFailure(noFailureTestingRandomSample, :);
testingLabels = [zeros(testingVectorSize,1); ones(testingVectorSize,1)]; % label "1" -> noFailure / positive

% ===== regularization of the training and testing set according to the training set

meanValues = mean([failureTrainingFeatureVector; noFailureTrainingFeatureVector;]);
standardDeviation = std([failureTrainingFeatureVector; noFailureTrainingFeatureVector;]);

stdNoFailureTrainingFeatureVector = (noFailureTrainingFeatureVector - meanValues) ./ standardDeviation;
stdFailureTrainingFeatureVector = (failureTrainingFeatureVector - meanValues) ./ standardDeviation;

stdNoFailureTestingFeatureVector = (noFailureTestingFeatureVector - meanValues) ./  standardDeviation;
stdFailureTestingFeatureVector = (failureTestingFeatureVector - meanValues) ./  standardDeviation;

% extract the initial signal number for each randomly selected signal in
% the testing set

counter = 1;
for i = 1:length(failureTestingRandomSample)
    if (failureTestingRandomSample(i) >= failureSignalIndex(counter) && failureTestingRandomSample(i) < failureSignalIndex(counter + 1))
        failureSignalNumber(i) = counter;
    else 
        %counter = counter + 1;
        while ~( failureTestingRandomSample(i) >= failureSignalIndex(counter + 1) && failureTestingRandomSample(i) < failureSignalIndex(counter + 2) )
            counter = counter + 1;
        end
        failureSignalNumber(i) = counter;        
    end
end


counter = 1;
for i = 1:length(noFailureTestingRandomSample)
    if (noFailureTestingRandomSample(i) >= noFailureSignalIndex(counter) && noFailureTestingRandomSample(i) < noFailureSignalIndex(counter + 1))
        noFailureSignalNumber(i) = counter;
    else 
        %counter = counter + 1;
        while ~( noFailureTestingRandomSample(i) >= noFailureSignalIndex(counter + 1) && noFailureTestingRandomSample(i) < noFailureSignalIndex(counter + 2) )
            counter = counter + 1;
        end
        noFailureSignalNumber(i) = counter;        
    end
end




clearvars -except stdFailureTrainingFeatureVector stdFailureTestingFeatureVector stdNoFailureTrainingFeatureVector stdNoFailureTestingFeatureVector feature_array_noFailure45min feature_array_failure45min trainingLabels testingLabels


%% extract testing set / process features

trainingVectorSize = 1000; % define size of training vector

% === choose training and testing set ===

feature_array_failure = feature_array_failure5min;
feature_array_noFailure = feature_array_noFailure5min;

feature_array_failure(isnan(feature_array_failure)) = 0; % remove 'NaN' and 'Inf' values should probably be removed
feature_array_noFailure(isnan(feature_array_noFailure)) = 0; 
feature_array_failure(isinf(feature_array_failure)) = 0;
feature_array_noFailure(isinf(feature_array_noFailure)) = 0;

% define training data
failureTrainingRandomSample = randsample(length(feature_array_failure), trainingVectorSize); % randomly choose class one (failure) samples for training
failureTrainingFeatureVector = feature_array_failure(failureTrainingRandomSample, :);
noFailureTrainingRandomSample = randsample(length(feature_array_noFailure), trainingVectorSize); % randomly choose class two (noFailure) sample for training
noFailureTrainingFeatureVector = feature_array_noFailure(noFailureTrainingRandomSample, :);
trainingLabels = [zeros(trainingVectorSize,1); ones(trainingVectorSize,1)]; % label = "0" -> failure / negative


% === define testing set

testingVectorSize = floor(0.01*length(failureSignalIndex5min));
failureTestingRandomSample = sort(randsample(length(failureSignalIndex5min), testingVectorSize)); % randomly choose class one (failure) samples for training
noFailureTestingRandomSample = sort(randsample(length(noFailureSignalIndex5min), testingVectorSize)); % randomly choose class one (failure) samples for training

failureTestingSignals = [];
noFailureTestingSignals = [];

for i = 1: length(failureTestingRandomSample)
    failureTestingSignals = [failureTestingSignals failure5min(:,failureSignalIndex5min(failureTestingRandomSample(i)) : failureSignalIndex5min(failureTestingRandomSample(i) + 1) -1 )];
    noFailureTestingSignals = [noFailureTestingSignals noFailure5min(:, noFailureSignalIndex5min(noFailureTestingRandomSample(i)) : noFailureSignalIndex5min(noFailureTestingRandomSample(i) + 1) -1 )];
end

testingVector = [failureTestingSignals noFailureTestingSignals];
testingLabels = [zeros(length(failureTestingSignals),1); ones(length(noFailureTestingSignals),1)]; % label "1" -> noFailure / positive
failureTestingFeatureVector = featExtractMod(failureTestingSignals);
noFailureTestingFeatureVector = featExtractMod(noFailureTestingSignals);

% ===== regularization of the testing set according to the training set

meanValues = mean([failureTrainingFeatureVector; noFailureTrainingFeatureVector;]);
standardDeviation = std([failureTrainingFeatureVector; noFailureTrainingFeatureVector;]);

stdNoFailureTrainingFeatureVector = (noFailureTrainingFeatureVector - meanValues) ./ standardDeviation;
stdFailureTrainingFeatureVector = (failureTrainingFeatureVector - meanValues) ./ standardDeviation;

stdNoFailureTestingFeatureVector = (noFailureTestingFeatureVector - meanValues) ./  standardDeviation;
stdFailureTestingFeatureVector = (failureTestingFeatureVector - meanValues) ./  standardDeviation;

% extract the initial signal number for each randomly selected signal in
% the testing set

counter = 1;
for i = 1:length(failureTestingFeatureVector)
    if (failureTestingFeatureVector(i) >= failureSignalIndex5min(counter) && failureTestingFeatureVector(i) < failureSignalIndex5min(counter + 1))
        failureSignalNumber(i) = counter;
    else 
        %counter = counter + 1;
        while ~( failureTestingFeatureVector(i) >= failureSignalIndex5min(counter + 1) && failureTestingFeatureVector(i) < failureSignalIndex5min(counter + 2) )
            counter = counter + 1;
        end
        failureSignalNumber(i) = counter;        
    end
end


counter = 1;
for i = 1:length(noFailureTestingFeatureVector)
    if (noFailureTestingFeatureVector(i) >= noFailureSignalIndex5min(counter) && noFailureTestingFeatureVector(i) < noFailureSignalIndex5min(counter + 1))
        noFailureSignalNumber(i) = counter;
    else 
        %counter = counter + 1;
        while ~( noFailureTestingFeatureVector(i) >= noFailureSignalIndex5min(counter + 1) && noFailureTestingFeatureVector(i) < noFailureSignalIndex5min(counter + 2) )
            counter = counter + 1;
        end
        noFailureSignalNumber(i) = counter;        
    end
end


%% train classifier 

classifier = 1; % choose classifier "1" for SVM

for j = 2:4
    j
    %allFeatVecs(:,j-1) = randsample(setdiff([1:24],[9,11,12,17,20,21]), 8); % choose feature vector for training out of 24 pre-calculated features
    %allFeatVecs(:,j-1) = randsample(24, 5); % choose feature vector for training out of 24 pre-calculated features
    FeatureVector = [1,2,3,4,5,7,8,19,18,17,24];

    %FeatureVector = allFeatVecs(:,j-1);
    trainFeat = [stdFailureTrainingFeatureVector(:, FeatureVector); stdNoFailureTrainingFeatureVector(:, FeatureVector)]; %final training feature vector combined
    testingFeat = [stdFailureTestingFeatureVector(:, FeatureVector); stdNoFailureTestingFeatureVector(:, FeatureVector)]; %final training feature vector combined

    for i = 1: 1
        i       
        [Accuracy(i,j-1), FalseNegatives(i,j), FalsePositives(i,j)] = learnAndTest(trainFeat, trainingLabels,testingFeat ,testingLabels, FeatureVector, 1);
    end

end

max = max(max(Accuracy))
min = min(min(Accuracy))

%% plot histograms

for i = 1:24
    figure
    histogram(stdFailureTrainingFeatureVector(:,i))
    hold on 
    histogram(stdNoFailureTrainingFeatureVector(:,i))
end

% plot signals befor failure

for i = 100:110
    hold on
    plot(normalOperationData{1, i})
end