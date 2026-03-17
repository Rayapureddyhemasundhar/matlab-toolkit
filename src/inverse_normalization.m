% ============================================================================
% INVERSE NORMALIZATION
% Reverse the transformation applied by safe_normalization
% ============================================================================

function original_data = inverse_normalization(scaled_data, scaler_params)
    % Recover original scale from normalized data.
    %
    % Inputs:
    %   scaled_data   - normalized matrix
    %   scaler_params - struct returned by safe_normalization

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
