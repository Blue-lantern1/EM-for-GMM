%% EE798L: Machine Learning for Wireless Communications
% MATLAB Assignment-3: EM for Gaussian mixture modelling
% NAME: S.Srikanth Reddy; Roll No: 22104092

clear all;
clc;
%% part1
mu1 = [3 3];
sigma1 = [1 0;0 2];

mu2 = [1 -3];
sigma2 = [2 0;0 1];

pi1=0.8; pi2=0.2; N=500;

R = zeros(N,2); % storing N points in R
zn1 = zeros(N,1); % hidden (latent) variables, can be used for comparision
zn2 = zeros(N,1); % with final assignments later on.
R1 = []; % for storing points of one cluster
R2 = []; % for storing points of other cluster

for i = 1:N
    if rand()<=pi1 % selecting cluster one with probability pi1
        R(i,1:2) = mvnrnd(mu1,sigma1);
        zn1(i) = 1;
        R1 = [R1;R(i,1:2)];
    else           % selecting cluster two with probability pi2
        R(i,1:2) = mvnrnd(mu2,sigma2);
        zn2(i) = 1;
        R2 = [R2;R(i,1:2)];
    end
    % we can use below 3 lines inside the loop to check how each point is
    % being generated using a breakpoint, rather than complete dataplot 
    % provided after the loop.
    %subplot(2,2,1)
    %plot(R(i,1),R(i,2),'.')
    %hold on;
end
subplot(2,2,1)
plot(R(:,1),R(:,2),'.')
title('Data Set')
xlabel('x_1')
ylabel('x_2')
%% part 2. & 3.
%random = rand();
%qn1_hat = random*ones(N,1);
%qn2_hat = (1-random)*ones(N,1);

% above 3 lines can be used to fix qnk's and calcuate parameters.
% since we can either fix parameters and calculate q (or) fix q and
% calculate parameters initially. Rest of the steps are continuation of 
% alternate maximization

pi1_hat = rand(); % initial pi1
pi2_hat = 1 - pi1_hat; % initial pi2

mu1_hat = [0 0]; % initial mu1
mu2_hat = [1 1]; % initial mu2

sigma1_hat = eye(2); % initial sigma1
sigma2_hat = eye(2); % initial sigma2

L(1) = -2; 
L(2) = -1;
t=2;
% above 3 lines for starting while loop
%%
while (abs(L(t)-L(t-1))>10^-12)
    % considered closeness of likelihood of current iteration w.r.t previous
    % iteration as stopping creteria.
    % Can use closeness of estimated parameters as well under certain
    % threshold as stopping creteria.
%%    
    p1l1 = pi1_hat*mvnpdf(R,mu1_hat,sigma1_hat); % calculating numerator of posterior 1
    p2l2 = pi2_hat*mvnpdf(R,mu2_hat,sigma2_hat); % calculating numerator of posterior 2

    qn1_hat = p1l1./(p1l1+p2l2); % calculating posterior 1
    qn2_hat = p2l2./(p1l1+p2l2); % calculating posterior 2
    
    pi1_hat = sum(qn1_hat)/N; % calculating pi1 using q1 calculated above
    pi2_hat = sum(qn2_hat)/N; % calculating pi2 using q2 calculated above

    mu1_hat = sum(qn1_hat.*R)/sum(qn1_hat); % calculating mu1 using q1 calculated above
    mu2_hat = sum(qn2_hat.*R)/sum(qn2_hat); % calculating mu2 using q2 calculated above

    sigma1_hat = (R-mu1_hat)'*diag(qn1_hat)*(R-mu1_hat)/sum(qn1_hat); % calculating sigma1 using q1 calculated above
    sigma2_hat = (R-mu2_hat)'*diag(qn2_hat)*(R-mu2_hat)/sum(qn2_hat); % calculating sigma2 using q2 calculated above

    t=t+1;
    L(t) = sum(log(p1l1 + p2l2)); % calculating Complete Data Likelihood
    B(t) = sum(qn1_hat*log(pi1_hat)+qn2_hat*log(pi2_hat)) + sum(qn1_hat.*(log(p1l1/pi1_hat)) + ...
        qn2_hat.*(log(p2l2/pi2_hat))) - sum(qn1_hat.*log(qn1_hat) + qn2_hat.*log(qn2_hat)); % calculating lower bound

end
%% Displaying parameter estimates
disp('cluster one');
disp('pi=');disp(pi1_hat);
disp('mean=');disp(mu1_hat');
disp('sigma=');disp(sigma1_hat);

disp('cluster two');
disp('pi=');disp(pi2_hat);
disp('mean=');disp(mu2_hat');
disp('sigma=');disp(sigma2_hat);
%% Comparing Likelihood and Lower Bound
subplot(2,2,2)
plot(L(3:end),'b') % excluding unrequired L(1) and L(2), actual calculations starts from t = 3
hold on;
plot(B(3:end),'g--') % excluding unrequired B(1) and B(2), actual calculations starts from t = 3
hold off;
title('Lower Bound convergence')
xlabel('iteration')
ylabel('L/B')
legend('Likelihood','Lower Bound','Location','best')
%% Data points assignment to clusters
r = qn1_hat./qn2_hat; % calculating ratio for soft decoding
asg1 = zeros(N,1); % for assinging to one of the clusters
asg2 = zeros(N,1); % for assinging to the other cluster
Clust1 = []; % for storing one cluster data
Clust2 = []; % for storing other cluster data
for i = 1:N
    if r(i) > 1
        asg1(i) = 1; % belongs to cluster with parameters pi1_hat, mu1_hat, sigma1_hat
        Clust1 = [Clust1;R(i,:)];
    else
        asg2(i) = 1; % belongs to cluster with parameters pi2_hat, mu2_hat, sigma2_hat
        Clust2 = [Clust2;R(i,:)];
    end
end
%% for contour plot
x1 = -3:0.1:6;
x2 = -6:0.1:9;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
R_1 = mvnpdf(X,mu1,sigma1);
R_1 = reshape(R_1,length(x2),length(x1)); % original parameters of one cluster
C1 = mvnpdf(X,mu1_hat,sigma1_hat);
C1 = reshape(C1,length(x2),length(x1)); % estimated parameters of one cluster

R_2 = mvnpdf(X,mu2,sigma2);
R_2 = reshape(R_2,length(x2),length(x1)); % original parameters of other cluster
C2 = mvnpdf(X,mu2_hat,sigma2_hat);
C2 = reshape(C2,length(x2),length(x1)); % estimated parameters of other cluster
%% Plotting clusters with original parameters
subplot(2,2,3)
plot(R1(:,1),R1(:,2),'o',R2(:,1),R2(:,2),'s')
hold on;
contour(x1,x2,R_1);
hold on;
contour(x1,x2,R_2)
hold off;
title('Origianl Data Clusters')
xlabel('x_1');
ylabel('x_2');
legend('\pi_1,\mu_1,\Sigma_1','\pi_2,\mu_2,\Sigma_2','Location','best')
%% Plotting clusters with estimated parameters
subplot(2,2,4)
plot(Clust1(:,1),Clust1(:,2),'o',Clust2(:,1),Clust2(:,2),'s')
hold on;
contour(x1,x2,C1)
hold on;
contour(x1,x2,C2)
hold off;
title('Simulated Data Clusters after assignment')
xlabel('x_1');
ylabel('x_2');
legend({'$\hat{\pi_1}$,$\hat{\mu_1}$,$\hat{\Sigma_1}$',['$\hat{\pi_2}$,$' ...
    '\hat{\mu_2}$,$\hat{\Sigma_2}$']},'Interpreter','latex','Location','best')