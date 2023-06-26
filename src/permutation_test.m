%% data preprocessing

% write function to search for the (r,s) pairs
addpath('../../data/')
% addpath('/home/shenggao/pnc/GCA/gca_code/TFOCS-master')
% addpath('/home/shenggao/pnc/GCA/gca_code/Official Code')
% addpath('/home/shenggao/pnc/experiment/cnb_demo_thick/cv_result/')
addpath('../sgca_code/')
addpath('../utils/TFOCS-master/')


% remove the missing values
demo = readtable('n1601_demographics_go1_20161212.csv');
% choose sex, ageAtScan1, handednessv2, fedu1, medu1
demo_age = demo(:,{'bblid','scanid','sex','ageAtScan1','medu1','handednessv2'});

%demo_age = demo(:,{'bblid','scanid','sex','ageAtScan1','handednessv2'});
demo_rm = rmmissing(demo_age);
thick = readtable('n1601_jlfAntsCTIntersectionCT_20170331.csv');
thick_rm = rmmissing(thick);
cnb =  readtable('n1601_cnb_zscores_all_fr_20161215.csv');
cnb_rm = rmmissing(cnb);


% join on bblid, scanid and split into three datasets
demo_name = demo_rm.Properties.VariableNames;
cnb_name  = cnb_rm.Properties.VariableNames;
thick_name = thick.Properties.VariableNames;

% average thick

tempjoin = innerjoin(cnb_rm,  demo_rm, 'Leftkeys', {'bblid', 'scanid'},'RightKeys',{'bblid','scanid'});
final_join = innerjoin(tempjoin, thick_rm, 'LeftKeys',{'bblid','scanid'},'RightKeys',{'bblid','scanid'});

total_name = final_join.Properties.VariableNames;
final_name = total_name(3:size(total_name,2));

% substract mean
cnb_cl =  final_join(:,cnb_name);
[~,cnb_size]  = size(cnb_cl);
cnb_mat = table2array(cnb_cl(:,3:cnb_size));
cnb_final  = cnb_mat - mean(cnb_mat);

demo_cl =  final_join(:,demo_name);
[~,demo_size]  = size(demo_cl);
demo_mat = table2array(demo_cl(:,3:demo_size));
demo_final  = demo_mat - mean(demo_mat);


thick_cl =  final_join(:,thick_name);
[~,thick_size]  = size(thick_cl);
thick_mat = table2array(thick_cl(:,3:thick_size));
thick_final  = thick_mat - mean(thick_mat);
left = 1:2:98;
right = 2:2:98;
thick_final(:,left) =0.5*(thick_final(:,left)+thick_final(:,right));
thick_final = thick_final(:,1:49); 

%deal with names
final_name = [final_name(1:30),final_name(31:2:128)];

% remove age, sex
final_name = [final_name(1:26),final_name(29:79)];

% calculate covariance and other parameters
p1 = cnb_size-2;
p2 = demo_size-4;
p3 = 0.5*(thick_size-2);
pp = [p1,p2,p3];
p =  sum(pp);
[n, ~] = size(cnb_final);

% scale by 12
demo_final_scale = demo_final;  
scale = 12; 
demo_final_scale(:,2) = demo_final(:,2)/scale;

% Perform residual (regress of other covariates on age, sex, age * sex)
X = cat(2, cnb_final, demo_final_scale, thick_final);
regX = X(:,27:28);
regX = [regX, regX(:,1) .* regX(:,2)];
regX = regX - mean(regX);
X = [X(:,1:26), X(:,29:79)];
beta = mvregress(regX,X);
R = X - regX * beta;
X = R;
X = normalize(X);

S = cov(X);
idx1 = 1:pp(1);
idx2 = pp(1)+1:pp(1)+pp(2);
idx3 = pp(1)+pp(2)+1:pp(1)+pp(2)+pp(3);
Mask = zeros(p);
Mask(idx1,idx1) = ones(pp(1),pp(1));
Mask(idx2,idx2) = ones(pp(2),pp(2));
Mask(idx3,idx3) = ones(pp(3),pp(3));
%final size of X is 888 * 80

Sigma0hat = S .* Mask;




%% Permutation Test
% Perform TGD
tol = 1e-6;
eta = 0.001;
lambda = 0.01;
max_iter = 20000;
best_rate = 0;


correct_rate = [];
scores = [];



T = 100;
train_score = zeros(2, T);
test_score = zeros(2, T);
A = X(:,idx1);
B = X(:,idx2);
C = X(:,idx3);
for i = 1:T
    % random permutation
    % B_perm = B(randperm(n),:);
    % C_perm = C(randperm(n),:);
    % X = cat(2, A, B_perm, C_perm);
    S = cov(X);
    Sigma0hat = S .* Mask;
    
    
    cv = cvpartition(size(X,1),'HoldOut',0.2);
    idx = cv.test;
    % Separate to training and test data
    Xtrain = X(~idx,:);
    Xtest  = X(idx,:);
    Strain = cov(Xtrain);
    Stest = cov(Xtest);
    S0train = Strain .* Mask;
    S0test = Stest .* Mask;

    %Perform Fantope on training data
    dir = 1;
    k1 = 5;
    [solx_train, solx1_train, SSS_train] = sgca_init(Xtrain, pp, dir, 0.5, 1e-6, 30);
    [aest_train, sest_train] = svd(solx_train);
    aest_train = aest_train(:,1:dir)*sest_train(1:dir,1:dir)^0.5;
    ainit_train = hard_thre(aest_train, k1);

    %perform TGD
    final_train = sgca_tgd(Xtrain, Strain,S0train,ainit_train,dir,k1,eta,lambda,tol,max_iter);
    a1_final = final_train;
    
    [P,~,~] = svd(Sigma0hat * a1_final, 'econ');
    X_new = X * (eye(p) - P*P');
    A = X_new(:,idx1);
    B = X_new(:,idx2);
    C = X_new(:,idx3);

    % Separate to training and test data
    Xtrain_new = X_new(~idx,:);
    Xtest_new  = X_new(idx,:);
    Strain_new = cov(Xtrain_new);
    Stest_new = cov(Xtest_new);
    S0train_new = Strain_new .* Mask;
    S0test_new = Stest_new .* Mask;


    
    % 2nd direction
    dir = 1;
    k2 = 10;
    % perform Fantope projection
    [solx_train, solx1_train, SSS_train] = sgca_init(Xtrain_new, pp, dir, 0.5, 1e-6, 30);
    [aest_train, sest_train] = svd(solx_train);
    aest_train = aest_train(:,1:dir)*sest_train(1:dir,1:dir)^0.5;
    
    ainit_train = hard_thre(aest_train, k2);
    
    %perform TGD
    final_train = sgca_tgd(Xtrain_new, Strain_new, Sigma0hat, ainit_train, dir, ...
                            k2,eta,lambda,tol,max_iter);
    a2_final = final_train;


    train_score(1, i) = trace(final_train' * Strain_new * final_train);
    test_score(1, i) = trace(final_train' * Stest_new * final_train);


    %permute and test
    A_perm = A(randperm(n),:);
    B_perm = B(randperm(n),:);
    C_perm = C(randperm(n),:);
    X_new = cat(2, A, B_perm, C_perm);


    Xtrain_new = X_new(~idx,:);
    Xtest_new  = X_new(idx,:);
    Strain_new = cov(Xtrain_new);
    Stest_new = cov(Xtest_new);
    S0train_new = Strain_new .* Mask;
    S0test_new = Stest_new .* Mask;


    
    % 2nd direction
    dir = 1;
    k2 = 10;
    % perform Fantope projection
    [solx_train, solx1_train, SSS_train] = sgca_init(Xtrain_new, pp, dir, 0.5, 1e-6, 30);
    [aest_train, sest_train] = svd(solx_train);
    aest_train = aest_train(:,1:dir)*sest_train(1:dir,1:dir)^0.5;
    
    ainit_train = hard_thre(aest_train, k2);
    
    %perform TGD
    final_train = sgca_tgd(Xtrain_new, Strain_new, Sigma0hat, ainit_train, dir, ...
                            k2,eta,lambda,tol,max_iter);
    a2_final = final_train;


    train_score(2, i) = trace(final_train' * Strain_new * final_train);
    test_score(2, i) = trace(final_train' * Stest_new * final_train);


    disp(i)


end



train_p = sum(train_score(2, :) > mean(train_score(1,: )))/T;
test_p = sum(test_score(2, :) > mean(train_score(1,: )))/T;
% pvalue = [train_p, test_p]
% save('rank2 pvalue','pvalue')




