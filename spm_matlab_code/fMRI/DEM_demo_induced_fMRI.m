% function DEM_demo_induced_fMRI
% Demonstration of DCM for CSD (fMRI) with simulated responses
%__________________________________________________________________________
% This demonstration compares generalised filtering and deterministic DCM 
% (generating complex cross spectra) in the context of a nonlinear 
% convolution (fMRI) model using simulated data. Here, the dynamic
% convolution model for fMRI responses is converted into a static
% non-linear model by generating not the timeseries per se but their
% second-order statistics - in the form of cross spectra and covariance
% functions. This enables model parameters to the estimated using the
% second order data features through minimisation of variational free
% energy. For comparison, the same data are inverted (in timeseries form)
% using generalised filtering. This example uses a particularly difficult
% problem - with limited data - to emphasise the differences.
%
% NB - the generalised filtering trakes much longer than the deterministic
% scheme
%__________________________________________________________________________
 
% Karl Friston
% Copyright (C) 2008-2022 Wellcome Centre for Human Neuroimaging
 
% Simulate timeseries
%==========================================================================
rng('default')
clear                 % DHedit
maxNumCompThreads(1)  % DHedit

% DEM Structure: create random inputs
% -------------------------------------------------------------------------
T  = 512;                             % number of observations (scans)
TR = 2;                               % repetition time or timing
t  = (1:T)*TR;                        % observation times
n  = 3;                               % number of regions or nodes
u  = spm_rand_mar(T,n,1/2)/4;         % endogenous fluctuations
 
% experimental inputs (Cu = 0 to suppress)
% -------------------------------------------------------------------------
% Cu  = [1; 0; 0] * 0;
Cu  = [1; zeros(n-1,1)] * 0;          % DHedit
E   = cos(2*pi*TR*(1:T)/24) * 0;

% priors
% -------------------------------------------------------------------------
options.nonlinear  = 0;
options.two_state  = 0;
options.stochastic = 1;
options.centre     = 1;
options.induced    = 1;
options.nograph    = 1;       % DHedit
options.maxnodes   = 50;      % DHedit, standard is 8 nodes, avoid the SVD trick.

 
A   = ones(n,n);
B   = zeros(n,n,0);
C   = zeros(n,n);
D   = zeros(n,n,0);
pP  = spm_dcm_fmri_priors(A,B,C,D,options);
 
 
% true parameters (reciprocal connectivity)
% -------------------------------------------------------------------------
% DHedit: for demo data we use the same connection matrix as was used in
% Friston et al. 2014.
% pP.A = [  0  -.2    0;
%          .4    0  -.3;
%           0   .2    0];

% DHedit: for speed comparison we draw random matrices:
tmp_foo = 0.1*randn(n,n);
tmp_foo = tmp_foo - diag(diag(tmp_foo));
pP.A = tmp_foo;

pP.C = eye(n,n);
pP.transit = randn(n,1)/16;
 
% simulate response to endogenous fluctuations
%==========================================================================
 
% integrate states
% -------------------------------------------------------------------------
M.f  = 'spm_fx_fmri';
M.x  = sparse(n,5);
U.u  = u + (Cu*E)';
U.dt = TR;
x    = spm_int_J(pP,M,U);
 
% haemodynamic observer
% -------------------------------------------------------------------------
for i = 1:T
    y(i,:) = spm_gx_fmri(spm_unvec(x(i,:),M.x),[],pP)';
end
 
% observation noise process
% -------------------------------------------------------------------------
e    = spm_rand_mar(T,n,1/2)/4;
 
% show simulated response
%--------------------------------------------------------------------------
% i = 1:128;
% spm_figure('Getwin','Figure 1'); clf
% subplot(2,2,1)
% plot(t(i),u(i,:))
% title('Endogenous fluctuations','FontSize',16)
% xlabel('Time (seconds)')
% ylabel('Amplitude')
% axis square
% 
% subplot(2,2,2), hold off
% plot(t(i),x(i,n + 1:end),'c'), hold on
% plot(t(i),x(i,1:n)), hold off
% title('Hidden states','FontSize',16)
% xlabel('Time (seconds)')
% ylabel('Amplitude')
% axis square
% 
% subplot(2,2,3)
% plot(t(i),y(i,:),t(i),e(i,:),':')
% title('Hemodynamic response and noise','FontSize',16)
% xlabel('Time (seconds)')
% ylabel('Amplitude')
% axis square
 
 
% nonlinear system identification (DCM for CSD)
%==========================================================================
DCM.options = options;
 
DCM.a    = logical(pP.A);
DCM.b    = zeros(n,n,0);
DCM.c    = logical(Cu);
DCM.d    = zeros(n,n,0);
 
% response
% -------------------------------------------------------------------------
DCM.Y.y  = y + e;
DCM.Y.dt = TR;
DCM.U.u  = E';
DCM.U.dt = TR;

 
% nonlinear system identification (Variational Laplace) - deterministic DCM
% =========================================================================
tstart = tic
% rng('default')
% DCM.M.pE.A = abs(0.01*randn(size(DCM.M.pE.A)));
CSD = spm_dcm_fmri_csd(DCM);
matcomptime = toc(tstart)
Ep = CSD.Ep;
Cp = full(CSD.Cp);
pE = CSD.M.pE;
pE.transit = full(pE.transit);
pE.decay = full(pE.decay);
pE.epsilon = full(pE.epsilon);
pE.a = full(pE.a);
pE.b = full(pE.b);
pE.c = full(pE.c);
pC = CSD.M.pC;
hE = CSD.M.hE;
ihC = CSD.M.hC^-1;
x = full(CSD.M.x);
F = CSD.F
dt = CSD.Y.dt;
Hz = CSD.Y.Hz;
data = CSD.Y.y;
csd = CSD.Y.csd;
true_params = pP;
% save(['/home/david/Projects/neuroblox/codes/Spectral-Dynamic-Causal-Modeling/speedandaccuracy/spm25_demo.mat'], 'F', 'matcomptime','Ep','Cp','pC','pE','hE','ihC','e','u','x','dt','Hz','data','csd','true_params')
% save(['/home/david/Projects/neuroblox/codes/SpectralDynamicCausalModeling/speed-comparison/fmri_' num2str(n) 'regions.mat'], 'F', 'matcomptime','Ep','Cp','pC','pE','hE','ihC','e','u','x','dt','Hz','data','csd','true_params')

% summary
% -------------------------------------------------------------------------
% spm_figure('Getwin','Figure 2'); clf
% 
% subplot(2,1,1); hold off
% spm_plot_ci(CSD.Ep.A(:),CSD.Cp(1:n*n,1:n*n)), hold on
% bar(pP.A(:),1/4), hold off
% title('True and MAP connections (Deterministic)','FontSize',16)
% axis square


% Bayesian deconvolution (Generalised filtering) - stochastic DCM
% =========================================================================

% initialise parameters using deterministic estimates
% -------------------------------------------------------------------------
% DCM.options.maxit = 8;
% DCM.options.pE    = rmfield(CSD.Ep,{'a','b','c'});
% 

% invert
% -------------------------------------------------------------------------
% LAP = spm_dcm_estimate(DCM);
% 
% % summary
% % -------------------------------------------------------------------------
% spm_figure('Getwin','Figure 2');
% 
% subplot(2,1,2); hold off
% spm_plot_ci(LAP.Ep.A(:),LAP.Cp(1:n*n,1:n*n)), hold on
% bar(pP.A(:),1/4), hold off
% title('True and MAP connections (Stochastic)','FontSize',16)
% axis square
% 

