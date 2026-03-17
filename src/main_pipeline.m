% ============================================================================
% MATLAB DATA PREPROCESSING & ANALYSIS TOOLKIT
% Main Pipeline Controller - Orchestrates entire workflow
% Author: Your Name
% Date: March 2026
% ============================================================================

function main_pipeline()

    clear; clc; close all;

    addpath(genpath('src'));
    addpath(genpath('utils'));

    cfg = config();

    if ~exist('logs', 'dir'), mkdir('logs'); end

    logger(repmat('=', 1, 60), 'INFO');
    logger('MATLAB DATA PREPROCESSING & ANALYSIS TOOLKIT', 'INFO');
    logger(repmat('=', 1, 60), 'INFO');
    logger(sprintf('Pipeline started at %s', datestr(now)), 'INFO');
    logger(sprintf('Configuration: dataset=%s, model=%s', cfg.dataset, cfg.ml_models{1}), 'INFO');

    % -------------------------------------------------------------------------
    % STEP 1: Load Dataset
    % -------------------------------------------------------------------------
    logger('STEP 1: Loading Dataset', 'INFO');
    logger(repmat('-', 1, 60), 'INFO');

    pipeline_start = tic;

    try
        [X, y, feature_names, dataset_info] = load_real_dataset(cfg.dataset, cfg.data_path);
        logger(sprintf('Dataset loaded: %s', dataset_info.name), 'INFO');
        logger(sprintf('  Samples:      %d', dataset_info.n_samples), 'INFO');
        logger(sprintf('  Features:     %d', dataset_info.n_features), 'INFO');
        logger(sprintf('  Missing rate: %.2f%%', 100 * dataset_info.missing_rate), 'INFO');
    catch ME
        logger(sprintf('Failed to load dataset: %s', ME.message), 'ERROR');
        rethrow(ME);
    end

    % -------------------------------------------------------------------------
    % STEP 2: Handle Missing Values
    % -------------------------------------------------------------------------
    logger('STEP 2: Missing Value Treatment', 'INFO');
    logger(repmat('-', 1, 60), 'INFO');

    missing_start = tic;
    [X_clean, imputation_results] = handle_missing_values_comprehensive(X, cfg);
    logger(sprintf('Missing values handled in %.2f seconds', toc(missing_start)), 'INFO');
    logger(sprintf('  Best method: %s', imputation_results.best_method), 'INFO');

    % -------------------------------------------------------------------------
    % STEP 3: Detect and Treat Outliers
    % -------------------------------------------------------------------------
    logger('STEP 3: Outlier Detection', 'INFO');
    logger(repmat('-', 1, 60), 'INFO');

    outlier_start = tic;
    [X_clean, outlier_results] = detect_outliers_comprehensive(X_clean, cfg);
    logger(sprintf('Outliers detected in %.2f seconds', toc(outlier_start)), 'INFO');
    logger(sprintf('  Outliers found: %d (%.2f%%)', ...
        outlier_results.total_outliers, 100 * outlier_results.outlier_rate), 'INFO');

    % -------------------------------------------------------------------------
    % STEP 4: Normalize Features
    % -------------------------------------------------------------------------
    logger('STEP 4: Feature Scaling', 'INFO');
    logger(repmat('-', 1, 60), 'INFO');

    norm_start = tic;
    [X_scaled, scaler_params] = safe_normalization(X_clean, 'standard');
    logger(sprintf('Features normalized in %.2f seconds', toc(norm_start)), 'INFO');
    logger('  Method: StandardScaler', 'INFO');

    % -------------------------------------------------------------------------
    % STEP 5: Feature Engineering
    % -------------------------------------------------------------------------
    logger('STEP 5: Feature Engineering', 'INFO');
    logger(repmat('-', 1, 60), 'INFO');

    feat_start = tic;
    [X_engineered, engineered_names] = engineer_features_comprehensive(X_scaled, feature_names, cfg);
    logger(sprintf('Features engineered in %.2f seconds', toc(feat_start)), 'INFO');
    logger(sprintf('  Original features:   %d', size(X_scaled, 2)), 'INFO');
    logger(sprintf('  Engineered features: %d', size(X_engineered, 2)), 'INFO');

    % -------------------------------------------------------------------------
    % STEP 6: Feature Selection
    % -------------------------------------------------------------------------
    logger('STEP 6: Feature Selection', 'INFO');
    logger(repmat('-', 1, 60), 'INFO');

    select_start = tic;
    [X_selected, selected_idx, selection_info] = feature_selection(X_engineered, y, cfg.feature_selection_method, cfg);
    logger(sprintf('Features selected in %.2f seconds', toc(select_start)), 'INFO');
    logger(sprintf('  Selected features: %d', size(X_selected, 2)), 'INFO');

    % -------------------------------------------------------------------------
    % STEP 7: Train/Test Split
    % -------------------------------------------------------------------------
    logger('STEP 7: Train/Test Split', 'INFO');
    logger(repmat('-', 1, 60), 'INFO');

    rng(cfg.random_seed);
    n        = size(X_selected, 1);
    n_train  = floor(cfg.train_ratio * n);
    idx      = randperm(n);

    X_train = X_selected(idx(1:n_train), :);
    y_train = y(idx(1:n_train));
    X_test  = X_selected(idx(n_train+1:end), :);
    y_test  = y(idx(n_train+1:end));

    logger(sprintf('  Training samples: %d', n_train), 'INFO');
    logger(sprintf('  Test samples:     %d', n - n_train), 'INFO');

    % -------------------------------------------------------------------------
    % STEP 8: Train Models
    % -------------------------------------------------------------------------
    logger('STEP 8: Model Training', 'INFO');
    logger(repmat('-', 1, 60), 'INFO');

    train_start = tic;
    ml_results = train_all_models(X_train, y_train, X_test, y_test, cfg);
    logger(sprintf('Models trained in %.2f seconds', toc(train_start)), 'INFO');

    % -------------------------------------------------------------------------
    % STEP 9: Hyperparameter Tuning (Optional)
    % -------------------------------------------------------------------------
    if cfg.hyperparameter_tuning
        logger('STEP 9: Hyperparameter Tuning', 'INFO');
        logger(repmat('-', 1, 60), 'INFO');

        tune_start = tic;
        [best_model, best_params, tuning_history] = hyperparameter_tuning(X_train, y_train, 'RandomForest', cfg);
        logger(sprintf('Hyperparameters tuned in %.2f seconds', toc(tune_start)), 'INFO');

        y_pred_tuned = predict(best_model, X_test);
        if iscell(y_pred_tuned)
            y_pred_tuned = str2double(y_pred_tuned);
        end
        rmse_tuned = sqrt(mean((y_test - y_pred_tuned).^2));
        logger(sprintf('  Tuned model RMSE: %.4f', rmse_tuned), 'INFO');
    end

    % -------------------------------------------------------------------------
    % STEP 10: Feature Importance Analysis
    % -------------------------------------------------------------------------
    logger('STEP 10: Feature Importance', 'INFO');
    logger(repmat('-', 1, 60), 'INFO');

    % Ensure we have a valid best model index and selected features
    best_model_idx = 1;
    if isfield(ml_results, 'rmse') && ~isempty(ml_results.rmse)
        [~, best_model_idx] = min(ml_results.rmse);
    end
    if isfield(ml_results, 'models') && ~isempty(ml_results.models)
        best_model_idx = min(max(best_model_idx, 1), numel(ml_results.models));
    else
        best_model_idx = 0;
    end

    if isempty(selected_idx) || any(selected_idx < 1) || any(selected_idx > numel(engineered_names))
        safe_selected_idx = 1:min(numel(engineered_names), size(X_engineered, 2));
    else
        safe_selected_idx = selected_idx;
    end

    try
        if best_model_idx > 0
            [importance_fig, top_features] = plot_feature_importance( ...
                ml_results.model_objects{best_model_idx}, ...
                ml_results.models{best_model_idx}, ...
                engineered_names(safe_selected_idx));
            if ~isempty(top_features)
                logger(sprintf('  Top feature: %s', top_features{1}), 'INFO');
            end
        else
            logger('  Skipping feature importance (no trained models available)', 'WARN');
        end
    catch ME
        logger(sprintf('Feature importance plot failed: %s', ME.message), 'WARN');
    end

    % -------------------------------------------------------------------------
    % STEP 11: Export Results
    % -------------------------------------------------------------------------
    logger('STEP 11: Exporting Results', 'INFO');
    logger(repmat('-', 1, 60), 'INFO');

    if ~exist('outputs', 'dir'), mkdir('outputs'); end
    export_cleaned_data(X_clean, X_engineered, X_selected, engineered_names(safe_selected_idx), y, ml_results);
    logger('Results exported to outputs/ directory', 'INFO');

    % -------------------------------------------------------------------------
    % STEP 12: Generate Report
    % -------------------------------------------------------------------------
    logger('STEP 12: Generating Report', 'INFO');
    logger(repmat('-', 1, 60), 'INFO');

    generate_pipeline_report(dataset_info, imputation_results, outlier_results, ...
        ml_results, selection_info, cfg);
    logger('HTML report generated: outputs/pipeline_report.html', 'INFO');

    % -------------------------------------------------------------------------
    % Pipeline Complete
    % -------------------------------------------------------------------------
    total_time = toc(pipeline_start);

    logger(repmat('=', 1, 60), 'INFO');
    logger('PIPELINE COMPLETED SUCCESSFULLY', 'INFO');
    logger(repmat('=', 1, 60), 'INFO');
    logger(sprintf('Total execution time: %.2f seconds', total_time), 'INFO');
    logger('Results saved in: outputs/', 'INFO');
    logger('Log saved in:     logs/pipeline.log', 'INFO');

    if ~exist('benchmarks', 'dir'), mkdir('benchmarks'); end
    save('benchmarks/benchmark_results.mat', 'total_time');

    fprintf('\n');
    fprintf('PIPELINE SUMMARY\n');
    fprintf('%s\n', repmat('-', 1, 50));
    fprintf('%-25s %s\n',  'Dataset:',            dataset_info.name);
    fprintf('%-25s %d\n',  'Original features:',  dataset_info.n_features);
    fprintf('%-25s %d\n',  'Engineered features:', size(X_engineered, 2));
    fprintf('%-25s %d\n',  'Selected features:',  size(X_selected, 2));
    if best_model_idx > 0 && best_model_idx <= numel(ml_results.models)
        fprintf('%-25s %s\n',  'Best model:',          ml_results.models{best_model_idx});
    fprintf('%-25s %.4f\n','Best RMSE:',           ml_results.rmse(best_model_idx));
    fprintf('%-25s %.4f\n','Best R2:',             ml_results.rsquared(best_model_idx));
    else
        fprintf('%-25s %s\n', 'Best model:', 'N/A');
        fprintf('%-25s %s\n', 'Best RMSE:',  'N/A');
        fprintf('%-25s %s\n', 'Best R2:',    'N/A');
    end
    fprintf('%s\n', repmat('-', 1, 50));

end
