% ============================================================================
% EXAMPLE RUNNER
% Minimal working demonstration of the preprocessing pipeline
% ============================================================================

function example_run()

    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('MATLAB DATA PREPROCESSING TOOLKIT - EXAMPLE RUN\n');
    fprintf('%s\n\n', repmat('=', 1, 60));

    addpath(genpath('src'));
    addpath(genpath('utils'));

    % Step 1: Configuration
    fprintf('[1/9] Loading configuration...\n');
    cfg         = config();
    cfg.dataset = 'synthetic';
    cfg.hyperparameter_tuning = false;

    % Step 2: Dataset
    fprintf('[2/9] Loading dataset...\n');
    [X, y, feature_names, info] = load_real_dataset(cfg.dataset);
    fprintf('      Loaded "%s" -- %d samples, %d features\n', ...
        info.name, info.n_samples, info.n_features);

    % Step 3: Missing values
    fprintf('[3/9] Handling missing values...\n');
    [X_clean, imp] = handle_missing_values_comprehensive(X, cfg);
    fprintf('      Best method: %s  (eval RMSE = %.4f)\n', imp.best_method, imp.best_rmse);

    % Step 4: Outlier detection
    fprintf('[4/9] Detecting and treating outliers...\n');
    [X_clean, out] = detect_outliers_comprehensive(X_clean, cfg);
    fprintf('      Consensus outliers: %d (%.2f%%)\n', ...
        out.total_outliers, 100*out.outlier_rate);

    % Step 5: Normalization
    fprintf('[5/9] Normalizing features...\n');
    [X_scaled, ~] = safe_normalization(X_clean, 'standard');
    fprintf('      Scaled range: [%.2f, %.2f]\n', min(X_scaled(:)), max(X_scaled(:)));

    % Step 6: Feature selection
    fprintf('[6/9] Selecting features...\n');
    [X_sel, sel_idx, sel_info] = feature_selection(X_scaled, y, 'pca', cfg);
    fprintf('      Selected %d of %d features\n', size(X_sel, 2), size(X_scaled, 2));

    % Step 7: Train / test split
    fprintf('[7/9] Splitting data...\n');
    rng(cfg.random_seed);
    n       = size(X_sel, 1);
    n_tr    = floor(cfg.train_ratio * n);
    idx     = randperm(n);
    X_train = X_sel(idx(1:n_tr), :);
    y_train = y(idx(1:n_tr));
    X_test  = X_sel(idx(n_tr+1:end), :);
    y_test  = y(idx(n_tr+1:end));
    fprintf('      Train: %d  Test: %d\n', n_tr, n - n_tr);

    % Step 8: Model training
    fprintf('[8/9] Training models...\n');
    ml = train_all_models(X_train, y_train, X_test, y_test, cfg);

    % Step 9: Results
    fprintf('[9/9] Exporting results...\n');
    if ~exist('outputs', 'dir'), mkdir('outputs'); end
    export_cleaned_data(X_clean, X_scaled, X_sel, feature_names(sel_idx), y, ml);

    [~, best] = min(ml.rmse);
    fprintf('\n%s\n', repmat('-', 1, 50));
    fprintf('Best model : %s\n', ml.models{best});
    fprintf('RMSE       : %.4f\n', ml.rmse(best));
    fprintf('R2         : %.4f\n', ml.rsquared(best));
    fprintf('MAE        : %.4f\n', ml.mae(best));
    fprintf('%s\n', repmat('-', 1, 50));
    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('EXAMPLE RUN COMPLETE\n');
    fprintf('%s\n', repmat('=', 1, 60));

end
