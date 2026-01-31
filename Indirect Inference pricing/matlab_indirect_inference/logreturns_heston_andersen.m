function [log_returns_trj, vola_trj] = logreturns_heston_andersen(mu, k, theta, sigma, rho, gamma1, gamma2, nu0, n_steps, dt, w1, w2)
% This function simulates the Heston model by Andersen QE (quadratic exponential) methodology

% mu = drift of the process
% k = speed of mean reversion of the process
% theta = long run volatility
% sigma = volatility of the stochastic vol process
% rho = correlations of between the two brownian motions
% X0 = initial value of the log asset
% nu0 = initial value of the volatility process
% n_steps = number of steps
% T = time horizon
% N = number of Monte-Carlo simulations
% w1 = uniform random numbers on [0,1]
% w2 = standard normal random numbers 

% dt = T/n_steps;
    
% uniform_sampling = unifrnd(0,1, n_steps-1, 2*n); % sampling numbers from continuous uniform distribution 
% The first n columns wil be used for the stochastic volatility  while the second 2 will be used for 
% the log-asset process

n = length(w1(1,1:end));

log_returns_trj = zeros(n_steps, 2*n); % Allocating the memory for the log-asset trajectory 
vola_trj = zeros(n_steps+1, 2*n); % allocating the memory for the volatility trajectory

%log_returns_trj(1, :) = params.X0; % fixing the initial value of the log-asset
vola_trj(1,:) = nu0; % fixing the initial value of the stochastic volatility 

u = [w1, 1 - w1];

% Numerical safety (avoid 0 or 1)
u = min(max(u, 1e-12), 1 - 1e-12);

gaussian_noise = sqrt(2) * erfinv(2*u - 1);


for j=2:n_steps

    m = cir_theo_mean(k, theta, vola_trj(j-1,:), dt); % theoretical mean of the CIR process
    s_square = cir_theo_variance(k, theta, sigma, vola_trj(j-1,:), dt); % theoretical variance of the CIR process
    
    psi = s_square./m.^2; % the discriminant for the approximation of the non central Chi-square distribution
    
    b_square = 2./psi - 1 + sqrt(2./psi) .* sqrt(2./psi - 1);
    a = m./(1+b_square);
    % gaussian_noise = icdf('Normal', [w1(j-1,1:n), 1-w1(j-1,1:n)], 0, 1);
    vola_gauss = a .* (sqrt(b_square) + gaussian_noise(j-1,1:end)).^2;

    p = (psi - 1)./(psi + 1);
    beta = (1-p)./m;

    inv_unif = 1./beta .* log((1-p)./(1 - [w1(j-1,1:n), 1-w1(j-1,1:n)]));
    vola_unif = max(0, inv_unif);
    
    vol = zeros(1,2*n);
    
%     if max(psi<=1.5)==1
%         vol(psi<=1.5) = vola_gauss(psi<=1.5);
%     end
% 
%     if max(psi>1.5)==1
%        vol(psi>1.5) = vola_unif(psi>1.5);
%     end

    vol(psi<=1.5) = vola_gauss(psi<=1.5);
    vol(psi>1.5) = vola_unif(psi>1.5);
    
    vola_trj(j,:) = vol;
    
end

K1 = gamma1 * dt * (k*rho/sigma -0.5) - rho/sigma;
K2 = gamma2 * dt * (k*rho/sigma -0.5) + rho/sigma;
K3 = gamma1 * dt * (1-rho^2);
K4 = gamma2 * dt * (1-rho^2);

K0 = -rho * k * theta /sigma * dt;

log_returns_trj(1:end,1:end) = mu*dt + K0 + ...
        K1.*vola_trj(1:end-1,1:end) + K2.*vola_trj(2:end,1:end) + ...
        sqrt(K3 .* vola_trj(1:end-1,1:end) + K4 .* vola_trj(2:end,1:end)) .* [w2, -w2];
end