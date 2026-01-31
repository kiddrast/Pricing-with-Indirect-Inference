%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Corsi & Renò indirect inference for fitting Heston with 1 factor LHAR-CJ as auxiliary model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Auxiliary model
fit.auxiliary_model = 'LHAR_CJ_TSRV';

%%%%%add stock name%%%%%


% Load real auxiliary estimates from Python
S = load(fullfile('C:\Users\alber\python_projects\fundamentals_of_interest_rates\Indirect Inference pricing\results\real_beta_vcv.mat'));
fit.n_days       = S.n_days;
%fit.n_days       = 30;
fit.coefficients = S.beta_real(:);
fit.VCV          = S.VCV_real;
fit.invVCV       = inv(fit.VCV);


% Input 
fit.n_intra = 80; % keep it like this, is like sampling every 5min (if 600 real obs per day)
fit.N       = 40;


% Seed
seed = 42;
rng(seed,'twister');
% CRN for smoother objective
w1 = double(rand(fit.n_days * fit.n_intra, fit.N));
w2 = double(randn(fit.n_days * fit.n_intra, fit.N));


% Optim options
fit.my_optim = optimset('Display','iter','TolFun',1e-2,'TolX',1e-4,'DiffMin',1e-4,'MaxFunEvals',500);
fit.par0 = [1.2, 1.0, 0.6, 0.0];
fit.LB   = [0.0, 0.0, 0.0, -1.0];
fit.UB   = [10.0, 2.0, 10.0,  1.0];
mu = 0;
dt = 1 / fit.n_intra;
eps = 1e-12;


% Optimizer (tic and toc measure the time)
tic
[a,fmin,ex] = fminsearchbnd3(@(par) minimum_LHARCJ_fit_one_factor_heston_andersen(fit.coefficients, par, mu, fit.invVCV, dt, fit.N, fit.n_intra, fit.n_days, w1, w2, eps),fit.par0, fit.LB, fit.UB, fit.my_optim);
toc


% Save results one folder up in \results 
savefile = fullfile('C:\Users\alber\python_projects\fundamentals_of_interest_rates\Indirect Inference pricing\results',['estimated_IF_', fit.auxiliary_model, '.mat']);

kappa = a(1);
theta = a(2);
sigma = a(3);
rho   = a(4);

save(savefile, 'kappa', 'theta', 'sigma', 'rho');
