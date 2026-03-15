% ============================================================================
% MISSING VALUE HANDLING MODULE
% Comprehensive missing value imputation with method evaluation
% ============================================================================

function [X_clean, results] = handle_missing_values_comprehensive(X, cfg)
    % Handle missing values using multiple imputation methods,
    % automatically selecting the one with the lowest reconstruction error.

    results               = struct();
    missing_mask          = isnan(X);
    results.total_missing = sum(missing_mask(:));
    results.missing_rate  = results.total_missing / numel(X);

    if results.total_missing == 0
        logger('  No missing values found.', 'INFO');
        X_clean              = X;
        results.best_method  = 'none';
        results.best_rmse    = 0;
        results.all_rmse     = [];
        return;
    end

    logger(sprintf('  Found %d missing values (%.2f%%)', ...
        results.total_missing, 100 * results.missing_rate), 'INFO');

    % Build evaluation set by artificially masking 10% of observed values
    eval_mask        = rand(size(X)) < 0.1 & ~missing_mask;
    original_values  = X(eval_mask);
    X_eval           = X;
    X_eval(eval_mask) = NaN;

    methods   = cfg.imputation_methods;
    n_methods = length(methods);
    rmse      = inf(n_methods, 1);

    for i = 1:n_methods
        method = methods{i};
        try
            X_filled      = apply_imputation(X_eval, method, cfg.imputation_k);
            imputed        = X_filled(eval_mask);
            rmse(i)        = sqrt(mean((imputed - original_values).^2));
        catch
            rmse(i) = inf;
        end
    end

    [min_rmse, best_idx]  = min(rmse);
    results.best_method   = methods{best_idx};
    results.best_rmse     = min_rmse;
    results.all_rmse      = rmse;

    logger(sprintf('  Best imputation method: %s (RMSE = %.4f)', ...
        results.best_method, min_rmse), 'INFO');

    X_clean = apply_imputation(X, results.best_method, cfg.imputation_k);

end


function X_filled = apply_imputation(X, method, k)
    % Dispatch to the appropriate imputation routine.
    switch method
        case {'linear', 'spline', 'pchip', 'previous', 'next', 'nearest'}
            X_filled = fillmissing(X, method);
        case 'mean'
            X_filled = fillmissing_by_mean(X);
        case 'median'
            X_filled = fillmissing_by_median(X);
        case 'knn'
            X_filled = fillmissing_by_knn(X, k);
        otherwise
            error('Unknown imputation method: %s', method);
    end
end


function X_filled = fillmissing_by_mean(X)
    X_filled = X;
    for col = 1:size(X, 2)
        col_mean = mean(X(:, col), 'omitnan');
        if isnan(col_mean), col_mean = 0; end
        X_filled(isnan(X(:, col)), col) = col_mean;
    end
end


function X_filled = fillmissing_by_median(X)
    X_filled = X;
    for col = 1:size(X, 2)
        col_median = median(X(:, col), 'omitnan');
        if isnan(col_median), col_median = 0; end
        X_filled(isnan(X(:, col)), col) = col_median;
    end
end


function X_filled = fillmissing_by_knn(X, k)
    X_filled  = X;
    n_features = size(X, 2);

    for col = 1:n_features
        missing  = isnan(X(:, col));
        if ~any(missing), continue; end

        other_cols = setdiff(1:n_features, col);
        complete   = ~any(isnan(X(:, other_cols)), 2) & ~missing;

        if sum(complete) > k
            X_train = X(complete, other_cols);
            y_train = X(complete, col);

            for row = find(missing)'
                if ~any(isnan(X(row, other_cols)))
                    distances   = sqrt(sum((X_train - X(row, other_cols)).^2, 2));
                    [~, idx]    = mink(distances, min(k, length(distances)));
                    X_filled(row, col) = mean(y_train(idx));
                else
                    X_filled(row, col) = mean(y_train);
                end
            end
        else
            col_mean = mean(X(~missing, col), 'omitnan');
            if isnan(col_mean), col_mean = 0; end
            X_filled(missing, col) = col_mean;
        end
    end
end
