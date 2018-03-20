%% load variables for labeling 
var1_1of2 = xlsread('MONSOON_POC2_d_2017_07.xlsx', 'B:B');
var1_2of2 = xlsread('MONSOON_POC2_d_2017_12.xlsx', 'B:B');
var1 = [var1_1of2; var1_2of2];
clear var1_1of2 var1_2of2;

var2_1of2 = xlsread('MONSOON_POC2_d_2017_07.xlsx', 'D:D');
var2_2of2 = xlsread('MONSOON_POC2_d_2017_12.xlsx', 'D:D');
var2 = [var2_1of2; var2_2of2];
clear var2_1of2 var2_2of2;

time1_1of2 = xlsread('MONSOON_POC2_d_2017_07.xlsx', 'A:A');
time1_2of2 = xlsread('MONSOON_POC2_d_2017_12.xlsx', 'A:A');
time1 = [time1_1of2; time1_2of2];
clear time1_1of2 time1_2of2;

time2_1of2 = xlsread('MONSOON_POC2_d_2017_07.xlsx', 'C:C');
time2_2of2 = xlsread('MONSOON_POC2_d_2017_12.xlsx', 'C:C');
time2 = [time2_1of2; time2_2of2];
clear time2_1of2 time2_2of2;

% failure: D110-J160_INT_MOY_MALAXEUR<200 ή D110-J010_VIT_MOT_VIS_DEM<50 : var1 ή D110-J050_DEB_INSTANTANE_DOSEUR<1 : var2.
var1Failure = find(var1 < 50);
var2Failure = find(var2 < 1);

plot(var1(var1Failure))
plot(var2(var2Failure))

labelFaultyVar1 = time1(var1Failure); % time stamps of failures due to var1
labelFaultyVar2 = time2(var2Failure); % time stamps of failures due to var2
commonLabel = unique([labelFaultyVar1; labelFaultyVar2]); % time stamps of failures


datestr(time1(i) + datenum('30-Dec-1899')); % number to date conversion



[num, txt] = xlsread('MONSOON_POC2_a_2017-07.xlsx', 'A:D');


