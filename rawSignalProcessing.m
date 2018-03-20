for i = 1 : 12
   signalFraction(:,i) = dataSignals{1,i};
end

labels = dataSignals{1,end}; % get labels
labels(labels~=-1) = 1; % change labels to binary (-1, 1) -> (normal, fault)

signalsLabels = [signalFraction labels];

ksdensity( signalsLabels((signalsLabels(:,end) == -1), 8 ))
hold on
ksdensity( signalsLabels((signalsLabels(:,end) == 1), 8 ))

figure 

for k = 1:13

    figure(k)
    ksdensity( signalsLabels((signalsLabels(:,end) == -1), k ))
    hold on
    ksdensity( signalsLabels((signalsLabels(:,end) == 1), k ))
end

for k = 1:13
    boxSignals = [signalsLabels((signalsLabels(:,end) == -1), k )' signalsLabels((signalsLabels(:,end) == 1), k )'];
    grp = [zeros(1, length(signalsLabels((signalsLabels(:,end) == -1), 1 )) ) , ones(1, length(signalsLabels((signalsLabels(:,end) == 1), 1 )) ) ];
    figure(k)
    boxplot(boxSignals, grp)
end


for k = 1 : 24
    failure = feature_array_failure45min(:,k);
    noFailure = feature_array_noFailure45min(:,k);
    boxSignals = [failure' noFailure'];
    grp = [zeros(1, length(failure) ) , ones(1, length(noFailure) ) ];
    
    figure
    subplot(2,1,1)
    boxplot(boxSignals, grp)
    
    subplot(2,1,2)
    ksdensity(failure)
    hold on
    ksdensity(noFailure)
end
