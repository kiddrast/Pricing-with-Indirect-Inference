function c = rbv_from_returns(r)
% Realized Bipower Variation (RBV)
% r must be intraday log-returns (vector) within a day.

r = r(:);

if length(r) < 2
    c = NaN;
    return;
end

c = (pi/2) * sum(abs(r(2:end)) .* abs(r(1:end-1)));
end
