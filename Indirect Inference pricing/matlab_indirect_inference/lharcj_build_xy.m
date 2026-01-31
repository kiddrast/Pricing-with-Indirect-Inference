function [y, X] = lharcj_build_xy(IV, C, r, eps)
% Build y and X for LHAR-CJ:
% y_t = log(IV_t + eps)
% X_t = [1, logC_d, logC_w, logC_m, log1pJ_d, r_d]
%
% IV, C, r are daily series (column vectors)

IV = IV(:);
C  = C(:);
r  = r(:);

T = length(IV);
if length(C) ~= T || length(r) ~= T
    error('IV, C, r must have same length.');
end

% Jump component
J = max(IV - C, 0);

% Lagged daily values
C_d = [NaN; C(1:end-1)];
J_d = [NaN; J(1:end-1)];
r_d = [NaN; r(1:end-1)];

% Weekly and monthly averages (STRICT min periods: 5 and 22)
C_w = NaN(T,1);
C_m = NaN(T,1);

for t = 1:T
    % C_w(t) = mean(C(t-5:t-1))
    if (t-1) >= 5
        C_w(t) = mean(C(t-5:t-1));
    end
    % C_m(t) = mean(C(t-22:t-1))
    if (t-1) >= 22
        C_m(t) = mean(C(t-22:t-1));
    end
end

% Dependent variable
y = log(IV + eps);

% Regressors
X = [ ...
    ones(T,1), ...
    log(C_d + eps), ...
    log(C_w + eps), ...
    log(C_m + eps), ...
    log(1 + max(J_d,0)), ...
    r_d ...
];

% Drop invalid rows
bad = any(isnan([y, X]), 2) | any(isinf([y, X]), 2);
y = y(~bad);
X = X(~bad,:);
end
