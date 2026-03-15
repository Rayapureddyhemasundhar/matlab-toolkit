% ============================================================================
% FEATURE ENGINEERING MODULE
% Creates a rich feature set from time-series and tabular data
% ============================================================================

function [X_engineered, feature_names] = engineer_features_comprehensive(X, original_names, cfg)
    % Build an expanded feature matrix from the input data.
    %
    % Constructed groups:
    %   1. Original features
    %   2. Rolling statistics (mean, std, min, max)
    %   3. Lag features
    %   4. First and second derivatives
    %   5. Cumulative mean and rolling range
    %   6. Ratio and product interaction features

    [n_samples, n_features] = size(X);
    X_engineered = [];
    feature_names = {};

    % Resolve original feature names
    orig = cell(1, n_features);
    for i = 1:n_features
        if ~isempty(original_names) && i <= length(original_names)
            orig{i} = original_names{i};
        else
            orig{i} = sprintf('Feature_%d', i);
        end
    end

    % -------------------------------------------------------------------------
    % Group 1: Original features
    % -------------------------------------------------------------------------
    X_engineered  = [X_engineered, X];
    feature_names = [feature_names, orig];

    % -------------------------------------------------------------------------
    % Group 2: Rolling statistics
    % -------------------------------------------------------------------------
    for w = cfg.rolling_windows
        if w >= n_samples, continue; end

        X_engineered  = [X_engineered, movmean(X, w, 'omitnan')];
        feature_names = [feature_names, append_suffix(orig, sprintf('_RMean%d', w))];

        X_engineered  = [X_engineered, movstd(X, w, 'omitnan')];
        feature_names = [feature_names, append_suffix(orig, sprintf('_RStd%d', w))];

        X_engineered  = [X_engineered, movmin(X, w, 'omitnan')];
        feature_names = [feature_names, append_suffix(orig, sprintf('_RMin%d', w))];

        X_engineered  = [X_engineered, movmax(X, w, 'omitnan')];
        feature_names = [feature_names, append_suffix(orig, sprintf('_RMax%d', w))];
    end

    % -------------------------------------------------------------------------
    % Group 3: Lag features
    % -------------------------------------------------------------------------
    for lag = cfg.lag_values
        if lag >= n_samples, continue; end
        lagged        = [NaN(lag, n_features); X(1:end-lag, :)];
        X_engineered  = [X_engineered, lagged];
        feature_names = [feature_names, append_suffix(orig, sprintf('_Lag%d', lag))];
    end

    % -------------------------------------------------------------------------
    % Group 4: Derivatives
    % -------------------------------------------------------------------------
    if cfg.include_derivatives
        diff1         = [NaN(1, n_features); diff(X)];
        X_engineered  = [X_engineered, diff1];
        feature_names = [feature_names, append_suffix(orig, '_D1')];

        diff2         = [NaN(2, n_features); diff(X, 2)];
        X_engineered  = [X_engineered, diff2];
        feature_names = [feature_names, append_suffix(orig, '_D2')];
    end

    % -------------------------------------------------------------------------
    % Group 5: Cumulative mean and range
    % -------------------------------------------------------------------------
    cummean       = cumsum(X) ./ (1:n_samples)';
    X_engineered  = [X_engineered, cummean];
    feature_names = [feature_names, append_suffix(orig, '_CumMean')];

    feat_range    = movmax(X, 50) - movmin(X, 50);
    X_engineered  = [X_engineered, feat_range];
    feature_names = [feature_names, append_suffix(orig, '_Range50')];

    % -------------------------------------------------------------------------
    % Group 6: Ratio and product interactions (first 5 features only)
    % -------------------------------------------------------------------------
    n_interact = min(n_features, 5);
    for i = 1:n_interact
        for j = i+1:n_interact
            if cfg.include_ratios
                ratio         = X(:, i) ./ (X(:, j) + eps);
                X_engineered  = [X_engineered, ratio];
                feature_names{end+1} = sprintf('Ratio_%s_%s', orig{i}, orig{j});
            end

            if cfg.include_products
                product       = X(:, i) .* X(:, j);
                X_engineered  = [X_engineered, product];
                feature_names{end+1} = sprintf('Prod_%s_%s', orig{i}, orig{j});
            end
        end
    end

    % Fill any residual NaN values
    X_engineered = fillmissing(X_engineered, 'linear', 'EndValues', 'nearest');

    n_new = size(X_engineered, 2) - n_features;
    logger(sprintf('  Added %d engineered features (%d total)', n_new, size(X_engineered, 2)), 'INFO');

end


function names = append_suffix(base_names, suffix)
    % Append a suffix string to every name in base_names.
    names = cellfun(@(n) [n, suffix], base_names, 'UniformOutput', false);
end
