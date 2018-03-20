signal16 = signalFraction(:,8);


colCounter = 1;
rowCounter = 1;

for i = 1:(length(signal16 )- 1) % check each whole feature signal
    
    % store signals from normal operation (label == -1)
    if ( labels(i) == -1 && (labels(i + 1) - labels(i)) == 0 ) % if current and previous label == -1
        signalOperationData{colCounter}(rowCounter,1) = signal16(i); % get all signals
        rowCounter = rowCounter + 1;
    elseif (labels(i) == -1 && (labels(i + 1) - labels(i)) ~= 0 ) % if label is not -1 change signal to be saved
        %changingPoint(colCounter) = i;
        colCounter = colCounter + 1;
        rowCounter = 1;
    end
    
end


clear colCounter rowCounter i counter j normalOperationData;

features = featExtract(signalOperationData); % extract features


