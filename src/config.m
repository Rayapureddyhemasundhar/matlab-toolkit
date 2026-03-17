% ============================================================================
% CONFIGURATION MODULE
% Central configuration for the entire pipeline
% ============================================================================

function cfg = config()

    % -------------------------------------------------------------------------
    % Data Configuration
    % -------------------------------------------------------------------------
    cfg.dataset     = 'air_quality';            % 'air_quality' | 'boston_housing' | 'weather' | 'stock' | 'synthetic'
    cfg.data_path   = 'data/air_quality.csv';   % Path to dataset
    cfg.train_ratio = 0.8;                      % Train/test split ratio
    cfg.random_seed = 42;                       % For reproducibility

    % -------------------------------------------------------------------------
    % Missing Value Configuration
    % -------------------------------------------------------------------------
    cfg.imputation_methods = {'linear', 'spline', 'pchip', 'previous', ...
                              'next', 'nearest', 'mean', 'median', 'knn'};
    cfg.imputation_k = 5;                       % K for KNN imputation

    % -------------------------------------------------------------------------
    % Outlier Detection Configuration
    % -------------------------------------------------------------------------
    cfg.outlier_methods        = {'zscore', 'modified_zscore', 'iqr', 'lof'};
    cfg.outlier_votes_required = 2;             % Consensus voting threshold
    cfg.zscore_threshold       = 3;             % Z-score threshold
    cfg.iqr_multiplier         = 1.5;           % IQR multiplier

    % -------------------------------------------------------------------------
    % Smoothing Configuration
    % -------------------------------------------------------------------------
    cfg.smoothing_windows = [5, 10, 20];        % Moving average windows
    cfg.smoothing_sigma   = [1, 2, 5];          % Gaussian sigma values
    cfg.smoothing_span    = [0.05, 0.1, 0.2];   % Lowess spans

    % -------------------------------------------------------------------------
    % Feature Engineering Configuration
    % -------------------------------------------------------------------------
    cfg.rolling_windows       = [5, 10, 20];    % Rolling statistics windows
    cfg.lag_values            = [1, 2, 3, 5];   % Lag features
    cfg.include_derivatives   = true;           % Include first/second derivatives
    cfg.include_ratios        = true;           % Include ratio features
    cfg.include_products      = true;           % Include product features

    % -------------------------------------------------------------------------
    % Feature Selection Configuration
    % -------------------------------------------------------------------------
    cfg.feature_selection_method = 'pca';       % 'pca' | 'correlation' | 'variance' | 'mutual_info' | 'rfe' | 'none'
    cfg.pca_variance_threshold   = 95;          % Keep 95% variance for PCA
    cfg.correlation_threshold    = 0.95;        % Remove features above this correlation

    % -------------------------------------------------------------------------
    % Machine Learning Configuration
    % -------------------------------------------------------------------------
    cfg.ml_models            = {'Linear', 'Ridge', 'Lasso', 'Tree', 'RandomForest', 'SVM', 'Ensemble'};
    cfg.cv_folds             = 5;               % Cross-validation folds
    cfg.hyperparameter_tuning = true;           % Enable hyperparameter optimization
    cfg.tuning_iterations    = 30;              % Number of optimization iterations

    % -------------------------------------------------------------------------
    % Visualization Configuration
    % -------------------------------------------------------------------------
    cfg.save_plots     = true;                  % Save figures to disk
    cfg.plot_format    = 'png';                 % 'png' | 'jpg' | 'fig'
    cfg.figure_width   = 1400;                  % Default figure width
    cfg.figure_height  = 900;                   % Default figure height

    % -------------------------------------------------------------------------
    % Logging Configuration
    % -------------------------------------------------------------------------
    cfg.log_level   = 'INFO';                   % 'DEBUG' | 'INFO' | 'WARN' | 'ERROR'
    cfg.log_to_file = true;                     % Write logs to file
    cfg.log_file    = 'logs/pipeline.log';      % Log file path

    % -------------------------------------------------------------------------
    % Export Configuration
    % -------------------------------------------------------------------------
    cfg.export_cleaned_data  = true;            % Export cleaned dataset
    cfg.export_model_results = true;            % Export model results
    cfg.export_report        = true;            % Generate HTML report

    % -------------------------------------------------------------------------
    % Create necessary directories
    % -------------------------------------------------------------------------
    directories = {'data', 'outputs', 'logs', 'benchmarks', 'figures'};
    for i = 1:length(directories)
        if ~exist(directories{i}, 'dir')
            mkdir(directories{i});
        end
    end

    % -------------------------------------------------------------------------
    % Add paths
    % -------------------------------------------------------------------------
    addpath(genpath('src'));
    addpath(genpath('utils'));

end
