% ============================================================================
% SAFE NORMALIZATION MODULE
% Robust feature scaling with division-by-zero protection
% ============================================================================

function [scaled_data, scaler_params] = safe_normalization(data, method)
    % Normalize data with protection against division by zero.
    %
    % Inputs:
    %   data   - numeric matrix [n_samples x n_features]
    %   method - 'standard' | 'minmax' | 'robust' | 'maxabs'
    %
    % Outputs:
    %   scaled_data   - normalized data
    %   scaler_params - parameters for inverse transform

    if nargin < 2
        method = 'standard';
    end

    logger(sprintf('Normalizing data using %s method', method), 'DEBUG');

    % Replace infinite values with NaN so they are handled by omitnan options
    data(~isfinite(data)) = NaN;

    scaled_data = zeros(size(data));
    scaler_params = struct();

    all_nan = all(isnan(data), 1);

    switch lower(method)

        case 'standard'
            mu    = mean(data, 1, 'omitnan');
            sigma = std(data,  0, 1, 'omitnan');

            sigma(sigma == 0 | isnan(sigma)) = 1;
            mu(isnan(mu)) = 0;

            scaled_data = (data - mu) ./ sigma;

            scaler_params.mu     = mu;
            scaler_params.sigma  = sigma;
            scaler_params.method = 'standard';

        case 'minmax'
            min_vals   = min(data, [], 1);
            max_vals   = max(data, [], 1);
            range_vals = max_vals - min_vals;

            range_vals(range_vals == 0 | isnan(range_vals)) = 1;
            min_vals(isnan(min_vals)) = 0;

            scaled_data = (data - min_vals) ./ range_vals;

            scaler_params.min_vals   = min_vals;
            scaler_params.range_vals = range_vals;
            scaler_params.method     = 'minmax';

        case 'robust'
            median_vals = median(data, 1, 'omitnan');
            iqr_vals    = iqr(data, 1);

            iqr_vals(iqr_vals == 0 | isnan(iqr_vals)) = 1;
            median_vals(isnan(median_vals)) = 0;

            scaled_data = (data - median_vals) ./ iqr_vals;

            scaler_params.median_vals = median_vals;
            scaler_params.iqr_vals    = iqr_vals;
            scaler_params.method      = 'robust';

        case 'maxabs'
            max_abs = max(abs(data), [], 1);
            max_abs(max_abs == 0 | isnan(max_abs)) = 1;

            scaled_data = data ./ max_abs;

            scaler_params.max_abs = max_abs;
            scaler_params.method  = 'maxabs';

        otherwise
            error('Unknown normalization method: %s', method);
    end

    scaled_data(~isfinite(scaled_data)) = 0;
    scaled_data(:, all_nan) = 0;

    logger(sprintf('Normalization complete. Output range: [%.2f, %.2f]', ...
        min(scaled_data(:)), max(scaled_data(:))), 'DEBUG');

end


function original_data = inverse_normalization(scaled_data, scaler_params)
    % Recover original scale from normalized data.

    switch scaler_params.method
        case 'standard'
            original_data = scaled_data .* scaler_params.sigma + scaler_params.mu;

        case 'minmax'
            original_data = scaled_data .* scaler_params.range_vals + scaler_params.min_vals;

        case 'robust'
            original_data = scaled_data .* scaler_params.iqr_vals + scaler_params.median_vals;

        case 'maxabs'
            original_data = scaled_data .* scaler_params.max_abs;

        otherwise
            error('Unknown scaler method: %s', scaler_params.method);
    end

end
