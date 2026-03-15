% ============================================================================
% MODEL TRAINING MODULE
% Trains and evaluates multiple regression models
% ============================================================================

function results = train_all_models(X_train, y_train, X_test, y_test, cfg)
    % Train all configured models and return a sorted comparison table.

    results             = struct();
    results.models      = {};
    results.rmse        = [];
    results.rsquared    = [];
    results.mae         = [];
    results.time        = [];
    results.predictions = {};

    for i = 1:length(cfg.ml_models)
        model_name = cfg.ml_models{i};
        logger(sprintf('  Training: %s', model_name), 'INFO');

        t_start = tic;
        try
            [model, y_pred] = fit_model(model_name, X_train, y_train, X_test);

            rmse = sqrt(mean((y_test - y_pred).^2));
            mae  = mean(abs(y_test - y_pred));
            ss_res = sum((y_test - y_pred).^2);
            ss_tot = sum((y_test - mean(y_test)).^2);
            r2   = 1 - ss_res / (ss_tot + eps);

            results.models{end+1}      = model_name;
            results.rmse(end+1)        = rmse;
            results.mae(end+1)         = mae;
            results.rsquared(end+1)    = r2;
            results.time(end+1)        = toc(t_start);
            results.predictions{end+1} = y_pred;

            logger(sprintf('    RMSE = %.4f  R2 = %.4f', rmse, r2), 'DEBUG');

        catch ME
            logger(sprintf('    Failed: %s', ME.message), 'WARN');
            results.models{end+1}      = model_name;
            results.rmse(end+1)        = inf;
            results.mae(end+1)         = inf;
            results.rsquared(end+1)    = -inf;
            results.time(end+1)        = toc(t_start);
            results.predictions{end+1} = zeros(size(y_test));
        end
    end

    % Sort by ascending RMSE
    [results.rmse, idx]    = sort(results.rmse);
    results.models         = results.models(idx);
    results.mae            = results.mae(idx);
    results.rsquared       = results.rsquared(idx);
    results.time           = results.time(idx);
    results.predictions    = results.predictions(idx);

    % Print comparison table
    fprintf('\nModel Performance Comparison:\n');
    fprintf('%s\n', repmat('-', 1, 72));
    fprintf('%-20s  %-10s  %-10s  %-10s  %-10s\n', 'Model', 'RMSE', 'MAE', 'R2', 'Time (s)');
    fprintf('%s\n', repmat('-', 1, 72));
    for i = 1:length(results.models)
        fprintf('%-20s  %-10.4f  %-10.4f  %-10.4f  %-10.4f\n', ...
            results.models{i}, results.rmse(i), results.mae(i), ...
            results.rsquared(i), results.time(i));
    end
    fprintf('%s\n', repmat('-', 1, 72));

end


function [model, y_pred] = fit_model(model_name, X_train, y_train, X_test)
    % Fit a single model and return predictions on the test set.

    switch lower(model_name)

        case 'linear'
            model  = fitlm(X_train, y_train);
            y_pred = predict(model, X_test);

        case 'ridge'
            model  = fitrlinear(X_train, y_train, ...
                'Learner', 'leastsquares', 'Regularization', 'ridge', 'Lambda', 0.1);
            y_pred = predict(model, X_test);

        case 'lasso'
            [B, fitinfo] = lasso(X_train, y_train, 'Lambda', 0.01);
            if isempty(B)
                y_pred = zeros(size(y_train));
            else
                y_pred = X_test * B + fitinfo.Intercept;
            end
            model  = struct('B', B, 'Intercept', fitinfo.Intercept);

        case 'tree'
            model  = fitrtree(X_train, y_train, 'MaxNumSplits', 20);
            y_pred = predict(model, X_test);

        case 'randomforest'
            model  = TreeBagger(50, X_train, y_train, ...
                'Method', 'regression', 'MinLeafSize', 5, 'OOBPrediction', 'on');
            y_pred = predict(model, X_test);
            if iscell(y_pred), y_pred = str2double(y_pred); end

        case 'svm'
            model  = fitrsvm(X_train, y_train, ...
                'KernelFunction', 'gaussian', 'Standardize', true);
            y_pred = predict(model, X_test);

        case 'ensemble'
            model  = fitrensemble(X_train, y_train, ...
                'Method', 'LSBoost', 'NumLearningCycles', 30);
            y_pred = predict(model, X_test);

        otherwise
            error('Unknown model type: %s', model_name);
    end

end
