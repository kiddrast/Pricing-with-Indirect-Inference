function iv = tsrv_from_logp(logp)
% TSRV from log-prices (vector). Two-Scales Realized Variance.
% logp must be a column vector of log prices within a day.

logp = logp(:);
n_points = length(logp);

if n_points < 3
    iv = NaN;
    return;
end

% Full-grid returns
r = diff(logp);
n = length(r);

% Subsampling frequency K ~ n^(2/3)
K = floor(n^(2/3));
K = max(1, min(K, n));

% RV on each subgrid
rv_k = zeros(K,1);
nret_k = zeros(K,1);

for k = 1:K
    idx = k:K:n_points;
    if length(idx) < 2
        rv_k(k) = 0;
        nret_k(k) = 0;
    else
        d = diff(logp(idx));
        rv_k(k) = sum(d.^2);
        nret_k(k) = length(idx) - 1;
    end
end

rv_avg = mean(rv_k);
nbar = mean(nret_k);

rv_all = sum(r.^2);

% TSRV (Zhang-Mykland-Ait-Sahalia eq. 55)
iv = rv_avg - (nbar/n) * rv_all;
end
