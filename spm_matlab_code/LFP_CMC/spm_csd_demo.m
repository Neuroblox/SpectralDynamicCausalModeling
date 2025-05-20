function spm_csd_demo(numsources)
% Demo routine for inverting local field potential models using
% cross-spectral density summaries of steady-state dynamics
%__________________________________________________________________________
% 
% This demo illustrates the inversion of neural-mass models (Moran et al
% 2005) of steady state-responses summarised in terms of the cross-spectral
% density. These data features are extracted using a vector
% auto-regression model and transformed into frequency space for subsequent
% inversion using a biophysical neural-mass model that is parameterised in
% terms of coupling and time constants.
%
% One can generate exemplar data by integrating the neural-mass model or by
% generating data directly from the cross-spectral DCM. In this demo we 
% use the former. DCM inversion using the standard nonlinear system 
% identification scheme spm_nlsi_N (a EM-like variational scheme under the 
% Laplace assumption).
% 
% NeuroImage. 2007 Sep 1;37(3):706-20.
% A neural mass model of spectral responses in electrophysiology.Moran RJ,
% Kiebel SJ, Stephan KE, Reilly RB, Daunizeau J, Friston KJ. 
%
% Abstract:
% We present a neural mass model of steady-state membrane potentials
% measured with local field potentials or electroencephalography in the
% frequency domain. This model is an extended version of previous dynamic
% causal models for investigating event-related potentials in the
% time-domain. In this paper, we augment the previous formulation with
% parameters that mediate spike-rate adaptation and recurrent intrinsic
% inhibitory connections. We then use linear systems analysis to show how
% the model's spectral response changes with its neurophysiological
% parameters. We demonstrate that much of the interesting behaviour depends
% on the non-linearity which couples mean membrane potential to mean
% spiking rate. This non-linearity is analogous, at the population level,
% to the firing rate-input curves often used to characterize single-cell
% responses. This function depends on the model's gain and adaptation
% currents which, neurobiologically, are influenced by the activity of
% modulatory neurotransmitters. The key contribution of this paper is to
% show how neuromodulatory effects can be modelled by adding adaptation
% currents to a simple phenomenological model of EEG. Critically, we show
% that these effects are expressed in a systematic way in the spectral
% density of EEG recordings. Inversion of the model, given such
% non-invasive recordings, should allow one to quantify pharmacologically
% induced changes in adaptation currents. In short, this work establishes a
% forward or generative model of electrophysiological recordings for
% psychopharmacological studies.
%__________________________________________________________________________
 
% Karl Friston
% Copyright (C) 2008-2022 Wellcome Centre for Human Neuroimaging
 
clear global
rng('default')        % DHedit
maxNumCompThreads(1)  % DHedit: ensure the use of only one 1 thread for speed comparison

 
% specify model
%==========================================================================
 
% number of sources and LFP channels (usually the same)
%--------------------------------------------------------------------------
n     = numsources;    % number of sources
nc    = n;             % number of channels
dipfit.Ns = n;         % DHedit
dipfit.Nc = nc;        % DHedit
dipfit.model = 'CMC';  % DHedit

 
% specify network (connections)            % DHedit: note that here we define masks not the actual connections
%--------------------------------------------------------------------------
% A{1}  = tril(ones(n,n),-1);              % a forward connection
A{1}  = tril(ones(n,n),-1);                % DHedit: a forward connection (sp -> ss and sp -> dp)
% A{2}  = triu(ones(n,n),+1);              % a backward connection
A{2}  = triu(ones(n,n),+1);                % DHedit: a backward connection (dp -> sp and dp -> ii)
% A{3}  = sparse(n,n);                     % lateral connections
B     = {};                                % trial-specific modulation
C     = speye(n,n);                        % sources receiving innovations
 
% get priors
%--------------------------------------------------------------------------
% [pE,pC] = spm_lfp_priors(A,B,C);              % neuronal priors
[pE,pC] = spm_dcm_neural_priors(A,B,C, 'CMC');  % neuronal priors, DHedit
% [pE,pC] = spm_L_priors(n,pE,pC);              % spatial  priors
[pE,pC] = spm_L_priors(dipfit,pE,pC);           % spatial priors, DHedit
[pE,pC] = spm_ssr_priors(pE,pC);                % spectral priors

% Suppress channel noise
%--------------------------------------------------------------------------
pE.b  = pE.b - 16;
pE.c  = pE.c - 16;
 
% create LFP model
%--------------------------------------------------------------------------
M.dipfit.type = 'LFP';

M.IS = 'spm_csd_mtf';
% M.FS = 'spm_fs_csd';           % DHedit: remove feature selection.
M.g  = 'spm_gx_erp';
% M.f  = 'spm_fx_lfp';
M.f  = 'spm_fx_cmc';             % DHedit
% M.x  = sparse(n,13);
M.x  = abs(0.0*randn(n, 8));     % DHedit: cmc has 8 states
% M.n  = n*13;
M.n  = n*8;                      % DHedit: cmc has 8 states
pE = rmfield(pE, 'Lpos');        % DHedit: remove some parameters to simplify the model as this is just a proof of principle
pE = rmfield(pE, 'G');           % DHedit
pE = rmfield(pE, 'R');           % DHedit
M.pE = pE;
pC.J = zeros(8,1);    % DAVID: remove this additional lead field terms and just keep superficial pyramidal
pC.C = zeros(size(pC.C));        % DHedit
pC = rmfield(pC,'Lpos');         % DHedit
pC = rmfield(pC,'G');            % DHedit
pC = rmfield(pC,'R');            % DHedit
M.pC = pC;
M.m  = n;
M.l  = nc;
M.Hz = (1:64)';
M.nograph = true;            % DHedit: avoid plotting, important for speed comparison
M.maxnodes = 100;            % DHedit, standard is 8 nodes, avoid the SVD trick and have results comparable to the Julia version

% simulate spectral data directly
%==========================================================================
P           = pE;
% P.A{1}(2,1) = 1/2;                        % strong forward connections
P.A{1}      = P.A{1} + A{1}.*randn(n);      % DHedit: forward connections (sp -> ss)
P.A{2}      = P.A{2} + A{1}.*randn(n);      % DHedit: forward connections (sp -> dp)
P.A{3}      = P.A{3} + A{2}.*randn(n);      % DHedit: backward connections (dp -> sp)
P.A{4}      = P.A{4} + A{2}.*randn(n);      % DHedit: backward connections (dp -> ii)
CSD         = spm_csd_mtf(P,M);
CSD         = CSD{1};
 
% or generate data and use the sample CSD
%==========================================================================
 
% Integrate with pink noise process
%--------------------------------------------------------------------------
N    = 512;
U.dt = 8/1000;
U.u  = randn(N,M.m)/16;
U.u  = sqrt(spm_Q(1/16,N))*U.u;
LFP  = spm_int_L(P,M,U);
 
% and estimate spectral features under a MAR model
%--------------------------------------------------------------------------
try
    mar = spm_mar(LFP, 8);
catch
    warndlg('please include spectral toolbax in Matlab path')
end
mar  = spm_mar_spectra(mar,M.Hz,1/U.dt);

% DHedit: don't plot for speed comparison
% spm_figure('GetWin','Figure 1'); clf
% 
% subplot(2,1,1)
% plot((1:N)*U.dt,LFP)
% xlabel('time')
% title('LFP')
% 
% subplot(2,1,2)
% plot(M.Hz,real(CSD(:,1,1)),M.Hz,real(CSD(:,1,2)),':')
% xlabel('frequency')
% title('[cross]-spectral density')
% axis square

 
% inversion (in frequency space)
%==========================================================================
 
% data and confounds
%--------------------------------------------------------------------------
Y.y   = {CSD};
 
% invert
%--------------------------------------------------------------------------
% Ep    = spm_nlsi_GN(M,[],Y);
% DHedit: setup speed comparison and store relevant variables
tstart = tic;
[Ep, Cp, ~, F]    = spm_nlsi_GN(M,[],Y);
matcomptime = toc(tstart);
x = full(M.x);
dt = 1;
Hz = M.Hz;
data = LFP;
csd = CSD;
true_params = P;
save(['~/Projects/neuroblox/codes/SpectralDynamicCausalModeling/speed-comparison/cmc_' num2str(n) 'regions.mat'], 'matcomptime','F','Ep','pE','Cp','pC','x','dt','Hz','data','csd','true_params')


% DHedit: don't plot for speed comparison

% spm_figure('GetWin','Figure 2'); clf
 
% plot spectral density
%==========================================================================
% [G w] = spm_csd_mtf(Ep,M);
 
% plot
%--------------------------------------------------------------------------
% g = G{1};
% y = Y.y{1};
% for i = 1:nc
%     for j = 1:nc
% 
%         subplot(3,2,(i - 1)*nc + j)
%         plot(w,real(g(:,i,j)),w,real(y(:,i,j)),':')
%         title(sprintf('cross-spectral density %d,%d',i,j))
%         xlabel('Power')
%         axis square
% 
%         try axis(a), catch, a = axis; end
% 
%     end
% end
% legend({'predicted','observed'})
% 
% % plot parameters and estimates
% %--------------------------------------------------------------------------
% subplot(3,2,5)
% bar(exp(spm_vec(P)))
% title('true parameters')
% 
% subplot(3,2,6)
% bar(exp(spm_vec(Ep)))
% title('conditional expectation')
