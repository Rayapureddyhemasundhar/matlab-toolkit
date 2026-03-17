% ============================================================================
% FEATURE SELECTION MODULE
% Multiple feature selection methods with automatic thresholding
% ============================================================================

function [X_selected, selected_features, selection_info] = feature_selection(X, y, method, cfg)
    % Select the most informative features using the specified method.
    %
    % Inputs:
    %   X      - feature matrix [n_samples x n_features]
    %   y      - target vector  [n_samples x 1]
    %   method - 'pca' | 'correlation' | 'variance' | 'mutual_info' | 'rfe' | 'none'
    %   cfg    - configuration struct

    if nargin < 4, cfg = config(); end
    if nargin < 3 || isempty(method), method = cfg.feature_selection_method; end

    n_original     = size(X, 2);
    selection_info = struct();
    selection_info.method     = method;
    selection_info.n_original = n_original;

    logger(sprintf('Performing feature selection: %s', method), 'INFO');

    if strcmpi(method, 'none')
        X_selected              = X;
        selected_features       = 1:n_original;
        selection_info.n_selected = n_original;
        logger('  No feature selection applied', 'INFO');
        return;
    end

    X_clean = fillmissing(X, 'linear');

    switch lower(method)
        case 'pca'
            [X_selected, selected_features, selection_info] = pca_selection(X_clean, cfg);

        case 'correlation'
            [X_selected, selected_features, selection_info] = correlation_selection(X_clean, y, cfg);

        case 'variance'
            [X_selected, selected_features, selection_info] = variance_selection(X_clean, cfg);

        case 'mutual_info'
            [X_selected, selected_features, selection_info] = mutual_info_selection(X_clean, y, cfg);

        case 'rfe'
            [X_selected, selected_features, selection_info] = rfe_selection(X_clean, y, cfg);

        otherwise
            error('Unknown feature selection method: %s', method);
    end

    selection_info.n_selected = size(X_selected, 2);
    logger(sprintf('Selected %d of %d features', size(X_selected, 2), n_original), 'INFO');

end


% -------------------------------------------------------------------------
% PCA
% -------------------------------------------------------------------------
function [X_selected, selected_features, info] = pca_selection(X, cfg)

    [coeff, score, ~, ~, explained] = pca(X, 'Centered', true);

    cumulative   = cumsum(explained);
    n_components = find(cumulative >= cfg.pca_variance_threshold, 1);
    if isempty(n_components)
        n_components = size(X, 2);
    end

    X_selected       = score(:, 1:n_components);
    selected_features = 1:n_components;

    info.method      = 'PCA';
    info.explained_variance = explained;
    info.cumulative  = cumulative;
    info.n_components = n_components;
    info.coeff       = coeff;

    if cfg.save_plots
        fig = figure('Visible', 'off');

        subplot(1, 2, 1);
        bar(explained(1:min(20, length(explained))));
        xlabel('Principal Component');
        ylabel('Explained Variance (%)');
        title('PCA Explained Variance');

        subplot(1, 2, 2);
        plot(cumulative, 'b-o', 'LineWidth', 2);
        hold on;
        yline(cfg.pca_variance_threshold, 'r--', 'LineWidth', 1.5);
        xlabel('Number of Components');
        ylabel('Cumulative Variance (%)');
        title('Cumulative Explained Variance');
        legend('Cumulative', sprintf('%d%% Threshold', cfg.pca_variance_threshold));

        fig_dir = fullfile(pwd, 'figures');
        if ~exist(fig_dir, 'dir')
            mkdir(fig_dir);
        end
        saveas(fig, fullfile(fig_dir, 'pca_analysis.png'));
        close(fig);
    end

end


% -------------------------------------------------------------------------
% Correlation Filter
% -------------------------------------------------------------------------
function [X_selected, selected_features, info] = correlation_selection(X, y, cfg)

    n_features  = size(X, 2);
    corr_matrix = corr(X, 'Rows', 'complete');
    high_corr   = abs(corr_matrix) > cfg.correlation_threshold;

    keep = true(1, n_features);
    for i = 1:n_features
        if keep(i)
            for j = i+1:n_features
                if high_corr(i, j)
                    keep(j) = false;
                end
            end
        end
    end

    if ~isempty(y)
        target_corr           = abs(corr(X, y, 'Rows', 'complete'));
        target_corr_threshold = prctile(target_corr, 75);
        keep                  = keep & (target_corr > target_corr_threshold)';
    end

    X_selected       = X(:, keep);
    selected_features = find(keep);

    info.method            = 'Correlation';
    info.kept              = keep;
    info.removed           = ~keep;
    info.correlation_matrix = corr_matrix;
    info.threshold         = cfg.correlation_threshold;

end


% -------------------------------------------------------------------------
% Variance Threshold
% -------------------------------------------------------------------------
function [X_selected, selected_features, info] = variance_selection(X, cfg)

    variances = var(X, 'omitnan');
    threshold = median(variances);
    keep      = variances > threshold;

    X_selected        = X(:, keep);
    selected_features = find(keep);

    info.method    = 'Variance';
    info.variances = variances;
    info.threshold = threshold;
    info.kept      = keep;

    if cfg.save_plots
        fig = figure('Visible', 'off');
        bar(sort(variances, 'descend'));
        hold on;
        yline(threshold, 'r--', 'LineWidth', 2);
        xlabel('Feature Index (sorted)');
        ylabel('Variance');
        title('Feature Variances');
        legend('Variance', 'Threshold');
        saveas(fig, 'figures/variance_analysis.png');
        close(fig);
    end

end


% -------------------------------------------------------------------------
% Mutual Information
% -------------------------------------------------------------------------
function [X_selected, selected_features, info] = mutual_info_selection(X, y, cfg)

    n_features = size(X, 2);
    mi_scores  = zeros(1, n_features);

    for i = 1:n_features
        mi_scores(i) = mutual_info(X(:, i), y);
    end

    n_keep                 = min(ceil(0.75 * n_features), 20);
    [sorted_scores, sort_idx] = sort(mi_scores, 'descend');
    keep                   = sort_idx(1:n_keep);

    X_selected        = X(:, keep);
    selected_features = keep;

    info.method        = 'Mutual Information';
    info.scores        = mi_scores;
    info.sorted_scores = sorted_scores;
    info.keep_idx      = keep;
    info.n_keep        = n_keep;

    if cfg.save_plots
        fig = figure('Visible', 'off');
        bar(mi_scores);
        hold on;
        bar(keep, mi_scores(keep), 'r');
        xlabel('Feature Index');
        ylabel('Mutual Information');
        title('Mutual Information Scores');
        legend('All Features', 'Selected');
        saveas(fig, 'figures/mutual_info.png');
        close(fig);
    end

end


% -------------------------------------------------------------------------
% Recursive Feature Elimination
% -------------------------------------------------------------------------
function [X_selected, selected_features, info] = rfe_selection(X, y, cfg)

    opts = statset('Display', 'off');
    [in, history] = sequentialfs(@rfe_criterion, X, y, ...
        'cv', cfg.cv_folds, 'options', opts, 'nfeatures', min(20, size(X, 2)));

    X_selected        = X(:, in);
    selected_features = find(in);

    info.method     = 'RFE';
    info.In         = in;
    info.history    = history;
    info.n_selected = sum(in);

end


% -------------------------------------------------------------------------
% Helper: Mutual Information
% -------------------------------------------------------------------------
function mi = mutual_info(x, y, n_bins)

    if nargin < 3, n_bins = 20; end

    valid = ~isnan(x) & ~isnan(y);
    x = x(valid);
    y = y(valid);

    if isempty(x) || numel(x) < 10
        mi = 0;
        return;
    end

    % Limit number of bins based on unique values to avoid empty bins
    n_bins = min(n_bins, max(2, numel(unique(x))));

    [~, x_edges] = histcounts(x, n_bins);
    [~, y_edges] = histcounts(y, n_bins);

    x_disc = discretize(x, x_edges);
    y_disc = discretize(y, y_edges);

    valid  = ~isnan(x_disc) & ~isnan(y_disc);
    x_disc = x_disc(valid);
    y_disc = y_disc(valid);

    if isempty(x_disc)
        mi = 0;
        return;
    end

    n     = length(x_disc);
    joint = accumarray([x_disc, y_disc], 1, [n_bins, n_bins]) / n;
    px    = sum(joint, 2);
    py    = sum(joint, 1)';

    mi = 0;
    for i = 1:n_bins
        for j = 1:n_bins
            if joint(i, j) > 0 && px(i) > 0 && py(j) > 0
                mi = mi + joint(i, j) * log(joint(i, j) / (px(i) * py(j) + eps));
            end
        end
    end

end


% -------------------------------------------------------------------------
% Helper: RFE criterion function
% -------------------------------------------------------------------------
function criterion = rfe_criterion(Xtrain, ytrain, Xtest, ytest)

    try
        model     = fitrtree(Xtrain, ytrain, 'MaxNumSplits', 10);
        ypred     = predict(model, Xtest);
        criterion = sqrt(mean((ytest - ypred).^2));
    catch
        criterion = inf;
    end

end
