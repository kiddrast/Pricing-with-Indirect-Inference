function [obj, betamean, stdbeta] = minimum_LHARCJ_fit_one_factor_heston_andersen(beta_real, par, mu, invVCV, dt, N, n_intra, n_days, w1, w2, eps)

% Parameters (Heston)
kappa = par(1);
theta = par(2);
sigma = par(3);
rho   = par(4);

beta_real = beta_real(:); % ensure column

k_beta = length(beta_real); % should be 6 if you match only LHAR-CJ betas
beta_diff = zeros(k_beta, 2*N);

% Simulate intraday log-returns (n_intra*n_days x 2N)
log_returns = logreturns_heston_andersen(mu, kappa, theta, sigma, rho, 0.5, 0.5, theta, n_intra*n_days, dt, w1, w2);

for j = 1:(2*N)

    rvec = log_returns(:, j);
    rmat = reshape(rvec, n_intra, n_days); % intraday x days

    % Daily series
    IV = NaN(n_days, 1);
    C  = NaN(n_days, 1);
    rd = NaN(n_days, 1);

    for d = 1:n_days
        r_day = rmat(:, d);

        % Daily return = sum intraday log-returns
        rd(d) = sum(r_day);

        % Build log-prices for TSRV
        logp_day = [0; cumsum(r_day)];

        IV(d) = tsrv_from_logp(logp_day);
        C(d)  = rbv_from_returns(r_day);
    end

    % Build LHAR-CJ regression and fit OLS
    [y, X] = lharcj_build_xy(IV, C, rd, eps);

    if length(y) < 50
        beta_diff(:, j) = NaN;
        continue;
    end

    beta_hat = X \ y; % OLS
    beta_diff(:, j) = beta_hat - beta_real;
end

% Remove simulations that produced NaNs (rare but possible if data is degenerate)
good = all(~isnan(beta_diff), 1);
beta_diff = beta_diff(:, good);

if isempty(beta_diff)
    obj = 1e12;
    betamean = NaN(k_beta,1);
    stdbeta  = NaN(k_beta,1);
    return;
end

betamean = mean(beta_diff, 2);
stdbeta  = std(beta_diff, 0, 2) / sqrt(size(beta_diff,2));

% Quadratic distance
obj = betamean' * invVCV * betamean;

end
