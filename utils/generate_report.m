% ============================================================================
% REPORT GENERATION MODULE
% Produces a self-contained HTML report of all pipeline results
% ============================================================================

function generate_pipeline_report(dataset_info, imputation_results, outlier_results, ...
                                   ml_results, selection_info, cfg)
    % Write an HTML report to outputs/pipeline_report.html.

    if ~exist('outputs', 'dir'), mkdir('outputs'); end
    filename = 'outputs/pipeline_report.html';
    fid = fopen(filename, 'w');

    [best_rmse, best_idx] = min(ml_results.rmse);

    % -------------------------------------------------------------------------
    % HTML head + CSS
    % -------------------------------------------------------------------------
    fprintf(fid, '<!DOCTYPE html>\n<html lang="en">\n<head>\n');
    fprintf(fid, '  <meta charset="UTF-8">\n');
    fprintf(fid, '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n');
    fprintf(fid, '  <title>Pipeline Report</title>\n');
    fprintf(fid, '  <style>\n');
    fprintf(fid, '    *{margin:0;padding:0;box-sizing:border-box}\n');
    fprintf(fid, '    body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;line-height:1.6;color:#333;background:#f4f6f8}\n');
    fprintf(fid, '    .container{max-width:1100px;margin:0 auto;padding:30px 20px}\n');
    fprintf(fid, '    h1{color:#1a1a2e;margin-bottom:6px;font-size:1.9rem}\n');
    fprintf(fid, '    .subtitle{color:#666;margin-bottom:30px;font-size:.9rem}\n');
    fprintf(fid, '    h2{color:#16213e;margin:36px 0 16px;padding-bottom:8px;border-bottom:2px solid #dee2e6;font-size:1.2rem}\n');
    fprintf(fid, '    .card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:20px 0}\n');
    fprintf(fid, '    .card{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:22px;border-radius:10px;text-align:center}\n');
    fprintf(fid, '    .card-value{font-size:2rem;font-weight:700;margin:6px 0}\n');
    fprintf(fid, '    .card-label{font-size:.8rem;opacity:.85;text-transform:uppercase;letter-spacing:.05em}\n');
    fprintf(fid, '    .card-sub{font-size:.78rem;opacity:.7;margin-top:4px}\n');
    fprintf(fid, '    .panel{background:#fff;border-radius:8px;padding:20px;margin:16px 0;box-shadow:0 1px 4px rgba(0,0,0,.08)}\n');
    fprintf(fid, '    table{width:100%%;border-collapse:collapse;font-size:.88rem}\n');
    fprintf(fid, '    th{background:#1a1a2e;color:#fff;padding:10px 14px;text-align:left;font-weight:600}\n');
    fprintf(fid, '    td{padding:10px 14px;border-bottom:1px solid #eee}\n');
    fprintf(fid, '    tr:last-child td{border-bottom:none}\n');
    fprintf(fid, '    tr:hover{background:#f8f9fa}\n');
    fprintf(fid, '    .best{background:#d4edda!important}\n');
    fprintf(fid, '    .badge{display:inline-block;padding:3px 8px;border-radius:4px;font-size:.78rem;font-weight:600}\n');
    fprintf(fid, '    .badge-ok{background:#28a745;color:#fff}\n');
    fprintf(fid, '    .badge-warn{background:#ffc107;color:#333}\n');
    fprintf(fid, '    .footer{margin-top:48px;padding-top:20px;border-top:1px solid #dee2e6;color:#888;font-size:.82rem;text-align:center}\n');
    fprintf(fid, '    @media(max-width:600px){.card-grid{grid-template-columns:1fr 1fr}}\n');
    fprintf(fid, '  </style>\n</head>\n<body>\n<div class="container">\n');

    % -------------------------------------------------------------------------
    % Header
    % -------------------------------------------------------------------------
    fprintf(fid, '<h1>Data Preprocessing Pipeline Report</h1>\n');
    fprintf(fid, '<p class="subtitle">Generated: %s</p>\n', datestr(now));

    % -------------------------------------------------------------------------
    % Summary cards
    % -------------------------------------------------------------------------
    fprintf(fid, '<div class="card-grid">\n');
    fprintf(fid, write_card(dataset_info.name, sprintf('%d samples', dataset_info.n_samples), 'Dataset'));
    fprintf(fid, write_card(ml_results.models{best_idx}, sprintf('RMSE %.4f', best_rmse), 'Best Model'));
    fprintf(fid, write_card(sprintf('%d &rarr; %d', dataset_info.n_features, selection_info.n_selected), 'features selected', 'Feature Count'));
    fprintf(fid, write_card(sprintf('%.1f%%', 100 * dataset_info.missing_rate), imputation_results.best_method, 'Missing Data'));
    fprintf(fid, '</div>\n');

    % -------------------------------------------------------------------------
    % Section 1: Dataset
    % -------------------------------------------------------------------------
    fprintf(fid, '<h2>1. Dataset Overview</h2>\n<div class="panel">\n');
    fprintf(fid, '<table>\n');
    fprintf(fid, '  <tr><th>Metric</th><th>Value</th></tr>\n');
    fprintf(fid, tr('Dataset', dataset_info.name));
    fprintf(fid, tr('Samples', num2str(dataset_info.n_samples)));
    fprintf(fid, tr('Original Features', num2str(dataset_info.n_features)));
    fprintf(fid, tr('Missing Values', sprintf('%d (%.2f%%)', dataset_info.n_missing, 100*dataset_info.missing_rate)));
    fprintf(fid, '</table>\n</div>\n');

    % -------------------------------------------------------------------------
    % Section 2: Imputation
    % -------------------------------------------------------------------------
    fprintf(fid, '<h2>2. Missing Value Treatment</h2>\n<div class="panel">\n');
    fprintf(fid, '<p style="margin-bottom:12px">Best method: <span class="badge badge-ok">%s</span> &nbsp; Evaluation RMSE: %.4f</p>\n', ...
        imputation_results.best_method, imputation_results.best_rmse);
    fprintf(fid, '<table>\n  <tr><th>Method</th><th>RMSE</th><th>Rank</th></tr>\n');

    [~, sorted_idx] = sort(imputation_results.all_rmse);
    for i = 1:length(sorted_idx)
        k      = sorted_idx(i);
        is_best = strcmp(cfg.imputation_methods{k}, imputation_results.best_method);
        row_cls = '';
        if is_best, row_cls = ' class="best"'; end
        fprintf(fid, '  <tr%s><td>%s</td><td>%.4f</td><td>#%d</td></tr>\n', ...
            row_cls, cfg.imputation_methods{k}, imputation_results.all_rmse(k), i);
    end
    fprintf(fid, '</table>\n</div>\n');

    % -------------------------------------------------------------------------
    % Section 3: Outlier Detection
    % -------------------------------------------------------------------------
    fprintf(fid, '<h2>3. Outlier Detection</h2>\n<div class="panel">\n');
    fprintf(fid, '<p style="margin-bottom:12px">Consensus outliers: <span class="badge badge-warn">%d (%.2f%%)</span></p>\n', ...
        outlier_results.total_outliers, 100 * outlier_results.outlier_rate);
    fprintf(fid, '<table>\n  <tr><th>Method</th><th>Outliers</th><th>%%</th></tr>\n');
    for i = 1:length(outlier_results.methods)
        fprintf(fid, '  <tr><td>%s</td><td>%d</td><td>%.2f</td></tr>\n', ...
            outlier_results.methods{i}, outlier_results.counts(i), outlier_results.percentages(i));
    end
    fprintf(fid, '</table>\n</div>\n');

    % -------------------------------------------------------------------------
    % Section 4: Feature Selection
    % -------------------------------------------------------------------------
    fprintf(fid, '<h2>4. Feature Selection</h2>\n<div class="panel">\n');
    reduction = 100 * (1 - selection_info.n_selected / selection_info.n_original);
    fprintf(fid, '<table>\n');
    fprintf(fid, tr('Method', upper(selection_info.method)));
    fprintf(fid, tr('Original Features', num2str(selection_info.n_original)));
    fprintf(fid, tr('Selected Features', num2str(selection_info.n_selected)));
    fprintf(fid, tr('Dimensionality Reduction', sprintf('%.1f%%', reduction)));
    fprintf(fid, '</table>\n</div>\n');

    % -------------------------------------------------------------------------
    % Section 5: Model Performance
    % -------------------------------------------------------------------------
    fprintf(fid, '<h2>5. Machine Learning Results</h2>\n<div class="panel">\n');
    fprintf(fid, '<table>\n  <tr><th>Model</th><th>RMSE</th><th>R&sup2;</th><th>MAE</th><th>Time (s)</th></tr>\n');
    for i = 1:length(ml_results.models)
        is_best = (i == 1);  % already sorted ascending
        row_cls = '';
        if is_best, row_cls = ' class="best"'; end
        label = ml_results.models{i};
        if is_best, label = [label, ' *']; end
        fprintf(fid, '  <tr%s><td>%s</td><td>%.4f</td><td>%.4f</td><td>%.4f</td><td>%.3f</td></tr>\n', ...
            row_cls, label, ml_results.rmse(i), ml_results.rsquared(i), ml_results.mae(i), ml_results.time(i));
    end
    fprintf(fid, '</table>\n<p style="font-size:.8rem;color:#666;margin-top:8px">* Best model</p>\n</div>\n');

    % -------------------------------------------------------------------------
    % Section 6: Configuration
    % -------------------------------------------------------------------------
    fprintf(fid, '<h2>6. Pipeline Configuration</h2>\n<div class="panel">\n<table>\n');
    fprintf(fid, tr('Train / Test Split', sprintf('%.0f%% / %.0f%%', cfg.train_ratio*100, (1-cfg.train_ratio)*100)));
    fprintf(fid, tr('Random Seed', num2str(cfg.random_seed)));
    fprintf(fid, tr('Cross-Validation Folds', num2str(cfg.cv_folds)));
    fprintf(fid, tr('Hyperparameter Tuning', mat2str(cfg.hyperparameter_tuning)));
    fprintf(fid, tr('Feature Selection Method', cfg.feature_selection_method));
    fprintf(fid, '</table>\n</div>\n');

    % -------------------------------------------------------------------------
    % Footer
    % -------------------------------------------------------------------------
    fprintf(fid, '<div class="footer">\n');
    fprintf(fid, '  <p>MATLAB Data Preprocessing Toolkit &mdash; outputs saved to <code>outputs/</code></p>\n');
    fprintf(fid, '  <p>Log file: <code>logs/pipeline.log</code></p>\n');
    fprintf(fid, '</div>\n</div>\n</body>\n</html>\n');

    fclose(fid);
    logger(sprintf('HTML report written to: %s', filename), 'INFO');

end


% -------------------------------------------------------------------------
% HTML helpers
% -------------------------------------------------------------------------

function s = write_card(value, sub, label)
    s = sprintf(['  <div class="card">\n',...
        '    <div class="card-label">%s</div>\n',...
        '    <div class="card-value">%s</div>\n',...
        '    <div class="card-sub">%s</div>\n',...
        '  </div>\n'], label, value, sub);
end


function s = tr(label, value)
    s = sprintf('  <tr><td>%s</td><td>%s</td></tr>\n', label, value);
end
