% ============================================================================
% EXPORT MODULE
% Saves processed data, feature matrices, and model results to disk
% ============================================================================

function export_cleaned_data(X_clean, X_engineered, X_selected, feature_names, y, ml_results)
    % Export all pipeline artefacts to the outputs/ directory.

    if ~exist('outputs', 'dir'), mkdir('outputs'); end

    % Cleaned data
    n_clean = size(X_clean, 2);
    clean_varnames = arrayfun(@(i) sprintf('Feature_%d', i), 1:n_clean, 'UniformOutput', false);
    T_clean = array2table(X_clean, 'VariableNames', clean_varnames);
    T_clean.Target = y;
    writetable(T_clean, 'outputs/cleaned_data.csv');
    logger('  Cleaned data   -> outputs/cleaned_data.csv', 'INFO');

    % Engineered features
    n_eng = size(X_engineered, 2);
    eng_varnames = arrayfun(@(i) sprintf('Eng_%d', i), 1:n_eng, 'UniformOutput', false);
    T_eng = array2table(X_engineered, 'VariableNames', eng_varnames);
    T_eng.Target = y;
    writetable(T_eng, 'outputs/engineered_features.csv');
    logger('  Engineered features -> outputs/engineered_features.csv', 'INFO');

    % Selected features
    if ~isempty(feature_names)
        T_sel = array2table(X_selected, 'VariableNames', feature_names);
        T_sel.Target = y;
        writetable(T_sel, 'outputs/selected_features.csv');
        logger('  Selected features  -> outputs/selected_features.csv', 'INFO');
    end

    % Model comparison table
    T_models = table( ...
        ml_results.models', ...
        ml_results.rmse', ...
        ml_results.rsquared', ...
        ml_results.mae', ...
        ml_results.time', ...
        'VariableNames', {'Model', 'RMSE', 'R2', 'MAE', 'Time_s'});
    writetable(T_models, 'outputs/model_comparison.csv');
    logger('  Model comparison   -> outputs/model_comparison.csv', 'INFO');

    % Full MATLAB workspace
    save('outputs/pipeline_results.mat', ...
        'X_clean', 'X_engineered', 'X_selected', 'y', 'ml_results', 'feature_names');
    logger('  Workspace          -> outputs/pipeline_results.mat', 'INFO');

end
