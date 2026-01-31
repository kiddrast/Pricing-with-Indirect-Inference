% This function computes the theoretical variance of the CIR model


% k = speed of mean reversion of the process
% theta = long run memory
% sigma = volatility of the process
% x0 = initial value
% t = starting point
% T = time horizon

function res = cir_theo_variance(k, theta, sigma, x0, dt)
res = x0 .* (sigma^2)/k .* (exp(-k.*dt) - exp(-2*k*dt)) + ((theta * sigma^2)/(2*k)) .* (1 - exp(-k.*dt)).^2;
end