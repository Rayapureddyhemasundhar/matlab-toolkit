% ============================================================================
% FEATURE IMPORTANCE MODULE
% Calculates and visualizes feature importance for trained models
% ============================================================================

function [fig, top_features] = plot_feature_importance(model, model_name, feature_names)
    % Plot feature importance for various model types.
    %
    % Inputs:
    %   model         - trained model object
    %   model_name    - string describing model type
    %   feature_names - cell array of feature names

    logger(sprintf('Calculating feature importance for %s', model_name), 'INFO');

    fig = figure('Position', [100, 100, 1200, 600], 'Visible', 'off');

    try
        % Extract importance based on model type
        if contains(model_name, 'RandomForest') || contains(model_name, 'TreeBagger')
            if isa(model, 'TreeBagger')
                importance = model.OOBPermutedPredictorDeltaError;
            else
                importance = predictorImportance(model);
            end
            title_str = 'Random Forest Feature Importance';

        elseif contains(model_name, 'Tree')
            importance = predictorImportance(model);
            title_str  = 'Decision Tree Feature Importance';

        elseif contains(model_name, 'Linear') || contains(model_name, 'Ridge') || contains(model_name, 'Lasso')
            if isa(model, 'LinearModel')
                importance = abs(model.Coefficients.Estimate(2:end));
            else
                importance = abs(model.Beta);
            end
            title_str = 'Linear Model Coefficient Magnitude';

        elseif contains(model_name, 'SVM')
            if strcmp(model.KernelFunction, 'linear')
                importance = abs(model.Beta);
                title_str  = 'SVM Feature Weights';
            else
                importance = ones(length(feature_names), 1);
                title_str  = 'SVM (non-linear) - Uniform Importance';
            end

        else
            importance = ones(length(feature_names), 1);
            title_str  = 'Feature Index (equal importance assumed)';
        end

        % Align importance vector length with feature names
        if length(importance) > length(feature_names)
            importance = importance(1:length(feature_names));
        elseif length(importance) < length(feature_names)
            importance = [importance; zeros(length(feature_names) - length(importance), 1)];
        end

        % Sort descending
        [sorted_importance, idx] = sort(importance, 'descend');
        sorted_names             = feature_names(idx);

        n_display        = min(15, length(sorted_names));
        sorted_importance = sorted_importance(1:n_display);
        sorted_names     = sorted_names(1:n_display);

        % Horizontal bar chart
        subplot(1, 2, 1);
        barh(sorted_importance(end:-1:1));
        set(gca, 'YTick', 1:n_display, 'YTickLabel', sorted_names(end:-1:1));
        xlabel('Importance');
        title(title_str);
        grid on;

        % Cumulative importance curve
        subplot(1, 2, 2);
        cumulative = cumsum(sorted_importance) / sum(sorted_importance) * 100;
        plot(1:n_display, cumulative, 'b-o', 'LineWidth', 2);
        hold on;
        yline(80, 'r--', 'LineWidth', 1.5);
        yline(95, 'g--', 'LineWidth', 1.5);
        xlabel('Number of Features');
        ylabel('Cumulative Importance (%)');
        title('Cumulative Feature Importance');
        legend('Cumulative', '80% Threshold', '95% Threshold', 'Location', 'best');
        grid on;

        sgtitle(sprintf('Feature Importance Analysis - %s', model_name));

        top_features = sorted_names(1:min(5, length(sorted_names)));

        if ~exist('figures', 'dir'), mkdir('figures'); end
        saveas(fig, sprintf('figures/feature_importance_%s.png', ...
            strrep(lower(model_name), ' ', '_')));

        fprintf('\nTop 5 Most Important Features:\n');
        for i = 1:min(5, length(top_features))
            fprintf('  %d. %s (importance: %.4f)\n', i, top_features{i}, sorted_importance(i));
        end

    catch ME
        logger(sprintf('Feature importance calculation failed: %s', ME.message), 'WARN');
        top_features = feature_names(1:min(5, length(feature_names)));

        subplot(1, 1, 1);
        bar(ones(length(feature_names), 1));
        xlabel('Feature Index');
        ylabel('Importance');
        title('Feature Importance (fallback - uniform)');
    end

    if nargout == 0
        set(fig, 'Visible', 'on');
    end

end
