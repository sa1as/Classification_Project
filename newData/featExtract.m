function feature_array = featExtract(dataVar)


% +----------------------------------------------------------------------+%
% |                                                                      |%
% |                   Offline Feature Extraction                         |%
% |                                                                      |%
% +----------------------------------------------------------------------+%

%dataVar = PlugData;
% -- Define Input Data --
%load('TASE_dataVar.mat');              % Load signal struct
%fields = fieldnames(dataVar);           % Return field names of a struct
%M = 24;                                 % Number of Features
%K = size(fields,1);                     % Number of Signals
%feature_array = zeros(K,M);             % Features [SignalNum, FeatureNum]



%dataVar = failure; %updatedNoFailure
M = 4;                                  % Number of Features
K = length(dataVar);                    % Number of Signals
feature_array = zeros(K,M);             % Features [SignalNum, FeatureNum]


% -- Extraction of Features on Time Field --

% -- 1. Absolute Values Sum --
for i=1:K
    Signal = dataVar(:,i);   % get one signal from struct
    feature_array(i,1)= sumabs(Signal);
end
1
% -- 2. Mean Absolute Value --
for i=1:K
    Signal = dataVar(:,i);
    N = length(Signal);
    feature_array(i,2)= feature_array(i,1)/N;
end
2

% -- 3. Modified Mean Absolute Value 1 --
for i=1:K
    Signal = dataVar(:,i);
    N = length(Signal);
    w = zeros(N,1);
    for j=1:N
        if (j>= 0.25*N && j <= 0.75*N)
            w(j) = 1;
        else
            w(j)= 0.5;
        end
    end
    feature_array(i,3)= sumabs(w.*Signal)/N;
end
3
% -- 4. Modified Mean Absolute Value 2 --
for i=1:K
    Signal = dataVar(:,i);
    N = length(Signal);
    w = zeros(N,1);
    for j=1:N
        if (j>= 0.25*N && j <= 0.75*N)
            w(j) = 1;
        elseif (0.25*N>j)
            w(j)= 4*j/N;
        else
            w(j)= 4*(j-N)/N;
        end
    end
    feature_array(i,4)= sumabs(w.*Signal)/N;
end
4
% -- 5. Simple Square Integral // Energy --
for i=1:K
    Signal = dataVar(:,i);
    feature_array(i,5)= sumabs(Signal.^2)/(length(Signal));
end
5
% -- 6. Variance --
for i=1:K
    Signal = dataVar(:,i);
    feature_array(i,6)= var(Signal);
end
6
% -- 7. Root Mean Square (RMS) --
for i=1:K
    Signal = dataVar(:,i);
    feature_array(i,7)= rms(Signal);%/length(Signal);
end
7
% -- 8. Waveform Length (WL) --
for i=1:K
    Signal = dataVar(:,i);
    N = length(Signal);
    s = zeros(N-1,1);
    for j=1:N-1
        s(j) = abs(Signal(j+1)-Signal(j));
    end
    feature_array(i,8)= sum(s)/N;         % Normalize by N
end
8
% -- 9. Hjorth Complexity --
for i=1:K
    Signal = dataVar(:,i);
    firstDeriv  = diff(Signal);
    secondDeriv = diff(firstDeriv);
    feature_array(i,9) = ((std(secondDeriv)*std(Signal))/(std(firstDeriv)*std(firstDeriv)))/length(Signal);
end
9
% -- 10. Shannon Entropy --
for i=1:K
    signal = dataVar(:,i);
    % -- Version 1 --
    %     clear p;
    %     a = unique(signal);
    %     out = [a,histc(signal(:),a)];
    %     for j = 1:length(signal)
    %         p(j) = out(find(out(:,1)==signal(j)),2)/length(signal);
    %     end
    
    % -- Version 2 --
    clear p;
    n_bins = 200;
    [pr, edges, h_bins] = histcounts(signal,n_bins);
    pr(pr(:) == 0) = 1;
    feature_array(i,10) = -sum((pr/sum(pr)).*log2((pr/sum(pr))))/log2(length(pr));
end
10
% -- 11. Skewness --
for i=1:K
    Signal = dataVar(:,i);
    feature_array(i,11) = skewness(Signal);%/length(Signal);
end
11
% -- 12. Kurtosis --
for i=1:K
    Signal = dataVar(:,i);
    feature_array(i,12) = kurtosis(Signal);%/length(Signal);
end
12
% -- 13. Mean Absolute Value Slope --
for i=1:K
    Signal = dataVar(:,i);
    N = length(Signal);
    feature_array(i,13) = ((sumabs(Signal(round(N/2):N))/size(Signal(round(N/2):N),1)) - (sumabs(Signal(1:round(N/2)-1))/size(Signal(1:round(N/2)-1),1)));%/length(Signal);
end
13
% -- 14 - 16. Histogram --
Nbins = 200;
first_N_bins = 3;
for i=1:K
    Signal = dataVar(:,i);
    counts = hist(Signal,Nbins);
    [counts,bins] = sort(counts,'descend');
    for j=1:first_N_bins
        feature_array(i,13+j) = bins(j);
    end
end
14-16
% -- 17. Hjorth Mobility --
for i=1:K
    Signal = dataVar(:,i);
    firstDeriv  = diff(Signal);
    feature_array(i,17) = sqrt(std(firstDeriv)/std(Signal))/length(Signal);
end
17
% -- 18. Maximum --
for i=1:K
    Signal = dataVar(:,i);
    feature_array(i,18) = max(Signal);
end
18
% -- 19. Log --
for i=1:K
    Signal = dataVar(:,i);
    feature_array(i,19) = exp(sum(log(abs(Signal)))/length(Signal));
end
19
% -- 20. Irregularity --
for i=1:K
    Signal = dataVar(:,i);
    s = Signal(1:length(Signal)-1) - Signal(2:length(Signal));
    s = vertcat(s,length(Signal)).^2;
    feature_array(i,20) = (sum(s)/sum(Signal.^2))/length(Signal);
end
20
% -- 21. Flatness --
for i=1:K
    Signal = dataVar(:,i);
    feature_array(i,21) = (geomean(abs(Signal))/mean(Signal))/length(Signal);
end
21
% -- 22. Difference absolute standard deviation value (DASDV) --
for i=1:K
    Signal = dataVar(:,i);
    N = length(Signal);
    s = zeros(N-1,1);
    for j=1:N-1
        s(j) = (Signal(j+1)-Signal(j))^2;
    end
    feature_array(i,22)= sqrt(sum(s)/(N-1));
end
22
% -- 23. Energy Percentage --
for i=1:K
    Signal = dataVar(:,i);
    N = length(Signal);
    over = 0;
    for j=1:N
        if Signal(j) > rms(Signal)
            over = over+1;
        end
    end
    feature_array(i,23)= over/N;
end
23
% -- 24. Willson Amplitude --
for i=1:K
    Signal = dataVar(:,i);
    N = length(Signal);
    f = zeros(N,1);
    for j=1:N-1
        if abs(Signal(j) - Signal(j+1)) > 0
            f(j) = 1;
        end
    end
    feature_array(i,24)= sum(f);%/N;
end






