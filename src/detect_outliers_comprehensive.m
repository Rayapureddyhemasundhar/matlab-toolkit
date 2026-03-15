% ============================================================================
% OUTLIER DETECTION MODULE
% Multiple outlier detection methods with consensus voting
% ============================================================================

function [X_clean, results] = detect_outliers_comprehensive(X, cfg)
    % Detect and treat outliers using five complementary methods.
    % Outliers are flagged by consensus vote; treatment uses winsorization.

    n_samples    = size(X, 1);
    outlier_votes = zeros(n_samples, 1);

    % -------------------------------------------------------------------------
    % Method 1: Z-Score
    % -------------------------------------------------------------------------
    z_scores  = abs((X - mean(X, 'omitnan')) ./ std(X, 'omitnan'));
    outliers_z = any(z_scores > cfg.zscore_threshold, 2);
    outlier_votes = outlier_votes + outliers_z;

    % -------------------------------------------------------------------------
    % Method 2: Modified Z-Score (median + MAD)
    % -------------------------------------------------------------------------
    median_vals = median(X, 'omitnan');
    mad_vals    = mad(X, 1);
    mad_vals(mad_vals == 0) = 1;
    modified_z  = 0.6745 * abs(X - median_vals) ./ mad_vals;
    outliers_mz  = any(modified_z > 3.5, 2);
    outlier_votes = outlier_votes + outliers_mz;

    % -------------------------------------------------------------------------
    % Method 3: IQR (Tukey's method)
    % -------------------------------------------------------------------------
    Q1          = prctile(X, 25);
    Q3          = prctile(X, 75);
    IQR_vals    = Q3 - Q1;
    outliers_iqr = any(X < Q1 - cfg.iqr_multiplier * IQR_vals | ...
                       X > Q3 + cfg.iqr_multiplier * IQR_vals, 2);
    outlier_votes = outlier_votes + outliers_iqr;

    % -------------------------------------------------------------------------
    % Method 4: Distance-based (PCA projection)
    % -------------------------------------------------------------------------
    try
        n_comp      = min(2, size(X, 2));
        coeff       = pca(X, 'NumComponents', n_comp);
        projected   = X * coeff;
        centroid    = mean(projected, 'omitnan');
        distances   = sqrt(sum((projected - centroid).^2, 2));
        threshold   = prctile(distances, 97.5);
        outliers_if = distances > threshold;
    catch
        outliers_if = false(n_samples, 1);
    end
    outlier_votes = outlier_votes + outliers_if;

    % -------------------------------------------------------------------------
    % Method 5: Local Outlier Factor (simplified)
    % -------------------------------------------------------------------------
    k           = min(20, floor(n_samples / 10));
    outliers_lof = detect_lof_outliers(X, k);
    outlier_votes = outlier_votes + outliers_lof;

    % -------------------------------------------------------------------------
    % Consensus voting
    % -------------------------------------------------------------------------
    results.methods         = {'Z-Score', 'Modified Z-Score', 'IQR', 'Distance-based', 'LOF'};
    results.votes           = outlier_votes;
    results.counts          = [sum(outliers_z), sum(outliers_mz), sum(outliers_iqr), ...
                                sum(outliers_if), sum(outliers_lof)];
    results.percentages     = 100 * results.counts / n_samples;
    results.consensus       = outlier_votes >= cfg.outlier_votes_required;
    results.total_outliers  = sum(results.consensus);
    results.outlier_rate    = results.total_outliers / n_samples;

    logger(sprintf('  Outliers detected: %d (%.2f%%) using consensus of %d methods', ...
        results.total_outliers, 100 * results.outlier_rate, cfg.outlier_votes_required), 'INFO');

    % -------------------------------------------------------------------------
    % Treatment: winsorization to 1st/99th percentile of inlier distribution
    % -------------------------------------------------------------------------
    X_clean = X;
    for col = 1:size(X, 2)
        col_data    = X(:, col);
        outliers_col = results.consensus & ~isnan(col_data);

        if any(outliers_col)
            inlier_data = col_data(~results.consensus);
            lower       = prctile(inlier_data, 1);
            upper       = prctile(inlier_data, 99);
            X_clean(outliers_col, col) = max(min(col_data(outliers_col), upper), lower);
        end
    end

    logger('  Outliers treated via winsorization (1st/99th percentile).', 'INFO');

end


function outliers = detect_lof_outliers(X, k)
    % Simplified LOF: flag points in the top 5% by LOF score.

    n        = size(X, 1);
    outliers = false(n, 1);

    if n < k * 2 || n > 5000
        return;
    end

    k_distances = zeros(n, 1);
    for i = 1:n
        distances        = sqrt(sum((X - X(i,:)).^2, 2));
        sorted_dist      = sort(distances);
        k_distances(i)   = sorted_dist(min(k+1, end));
    end

    lof_scores = zeros(n, 1);
    for i = 1:n
        distances  = sqrt(sum((X - X(i,:)).^2, 2));
        [~, idx]   = sort(distances);
        neighbors  = idx(2:min(k+1, end));

        reach_dist = max(k_distances(neighbors), distances(neighbors));
        lrd_i      = 1 / (mean(reach_dist) + eps);

        lrd_nb = zeros(length(neighbors), 1);
        for j = 1:length(neighbors)
            nb         = neighbors(j);
            d_nb       = sqrt(sum((X - X(nb,:)).^2, 2));
            [~, nb_idx] = sort(d_nb);
            nb_nb      = nb_idx(2:min(k+1, end));
            rd_nb      = max(k_distances(nb_nb), d_nb(nb_nb));
            lrd_nb(j)  = 1 / (mean(rd_nb) + eps);
        end

        lof_scores(i) = mean(lrd_nb) / (lrd_i + eps);
    end

    threshold = prctile(lof_scores, 95);
    outliers  = lof_scores > threshold;

end
