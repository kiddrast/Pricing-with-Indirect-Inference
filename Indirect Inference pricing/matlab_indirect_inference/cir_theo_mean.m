
% This function computes the theoretical mean of the cir process


% k = speed of mean reversion of the process
% theta = long run memory
% sigma = volatility of the process
% x0 = initial value
% t = starting point
% T = time horizon

function res = cir_theo_mean(k, theta, x0, dt)
res = x0 .* exp(-k*dt) + theta .* (1 - exp(-k*dt));
end