% ============================================================================
% HYPERPARAMETER TUNING MODULE
% Bayesian optimization for ML model hyperparameters
% ============================================================================

function [best_model, best_params, tuning_history] = hyperparameter_tuning(X_train, y_train, model_type, cfg)
    % Tune hyperparameters for the specified model type.
    %
    % Supported types: 'Ridge', 'Lasso', 'SVM', 'Tree', 'RandomForest', 'Ensemble'

    logger(sprintf('Tuning hyperparameters for: %s', model_type), 'INFO');

    switch lower(model_type)
        case 'ridge'
            [best_model, best_params, tuning_history] = tune_ridge(X_train, y_train, cfg);

        case 'lasso'
            [best_model, best_params, tuning_history] = tune_lasso(X_train, y_train, cfg);

        case 'svm'
            [best_model, best_params, tuning_history] = tune_svm(X_train, y_train, cfg);

        case 'tree'
            [best_model, best_params, tuning_history] = tune_tree(X_train, y_train, cfg);

        case 'randomforest'
            [best_model, best_params, tuning_history] = tune_random_forest(X_train, y_train, cfg);

        case 'ensemble'
            [best_model, best_params, tuning_history] = tune_ensemble(X_train, y_train, cfg);

        otherwise
            error('Unknown model type for tuning: %s', model_type);
    end

    logger(sprintf('Tuning complete for %s.', model_type), 'INFO');

end


% -------------------------------------------------------------------------
% Per-model tuning routines
% -------------------------------------------------------------------------

function [best_model, best_params, history] = tune_ridge(X, y, cfg)
    obj_fn = @(p) cv_error_ridge(X, y, p.lambda, cfg.cv_folds);
    params = optimizableVariable('lambda', [1e-5, 1e3], 'Transform', 'log');
    results = bayesopt(obj_fn, params, ...
        'MaxObjectiveEvaluations', cfg.tuning_iterations, 'Verbose', 0, ...
        'AcquisitionFunctionName', 'expected-improvement-plus');
    best_params.lambda = results.XAtMinObjective.lambda;
    best_model = fitrlinear(X, y, 'Learner', 'leastsquares', ...
        'Regularization', 'ridge', 'Lambda', best_params.lambda);
    history = results;
end


function [best_model, best_params, history] = tune_lasso(X, y, cfg)
    obj_fn = @(p) cv_error_lasso(X, y, p.lambda, cfg.cv_folds);
    params = optimizableVariable('lambda', [1e-5, 1e-1], 'Transform', 'log');
    results = bayesopt(obj_fn, params, ...
        'MaxObjectiveEvaluations', cfg.tuning_iterations, 'Verbose', 0);
    best_params.lambda = results.XAtMinObjective.lambda;
    [B, fitinfo] = lasso(X, y, 'Lambda', best_params.lambda);
    best_model = struct('B', B, 'Intercept', fitinfo.Intercept);
    history = results;
end


function [best_model, best_params, history] = tune_svm(X, y, cfg)
    obj_fn = @(p) cv_error_svm(X, y, p.C, p.sigma, p.epsilon, cfg.cv_folds);
    params = [
        optimizableVariable('C',       [1e-3, 1e3],  'Transform', 'log')
        optimizableVariable('sigma',   [0.1,  10],   'Transform', 'log')
        optimizableVariable('epsilon', [1e-4, 1],    'Transform', 'log')
    ];
    results = bayesopt(obj_fn, params, ...
        'MaxObjectiveEvaluations', cfg.tuning_iterations, 'Verbose', 0);
    best_params.box_constraint = results.XAtMinObjective.C;
    best_params.kernel_scale   = results.XAtMinObjective.sigma;
    best_params.epsilon        = results.XAtMinObjective.epsilon;
    best_model = fitrsvm(X, y, 'KernelFunction', 'gaussian', ...
        'BoxConstraint', best_params.box_constraint, ...
        'KernelScale',   best_params.kernel_scale, ...
        'Epsilon',       best_params.epsilon, ...
        'Standardize',   true);
    history = results;
end


function [best_model, best_params, history] = tune_tree(X, y, cfg)
    obj_fn = @(p) cv_error_tree(X, y, p.min_leaf, p.max_splits, cfg.cv_folds);
    params = [
        optimizableVariable('min_leaf',  [1,  50],  'Type', 'integer')
        optimizableVariable('max_splits',[1,  100], 'Type', 'integer')
    ];
    results = bayesopt(obj_fn, params, ...
        'MaxObjectiveEvaluations', cfg.tuning_iterations, 'Verbose', 0);
    best_params.min_leaf   = results.XAtMinObjective.min_leaf;
    best_params.max_splits = results.XAtMinObjective.max_splits;
    best_model = fitrtree(X, y, ...
        'MinLeafSize', best_params.min_leaf, ...
        'MaxNumSplits', best_params.max_splits);
    history = results;
end


function [best_model, best_params, history] = tune_random_forest(X, y, cfg)
    obj_fn = @(p) cv_error_rf(X, y, p.num_trees, p.min_leaf, cfg.cv_folds);
    params = [
        optimizableVariable('num_trees', [10,  200], 'Type', 'integer')
        optimizableVariable('min_leaf',  [1,   50],  'Type', 'integer')
    ];
    results = bayesopt(obj_fn, params, ...
        'MaxObjectiveEvaluations', cfg.tuning_iterations, 'Verbose', 0);
    best_params.num_trees = results.XAtMinObjective.num_trees;
    best_params.min_leaf  = results.XAtMinObjective.min_leaf;
    best_model = TreeBagger(best_params.num_trees, X, y, ...
        'Method', 'regression', 'MinLeafSize', best_params.min_leaf, ...
        'OOBPrediction', 'on');
    history = results;
end


function [best_model, best_params, history] = tune_ensemble(X, y, cfg)
    obj_fn = @(p) cv_error_ensemble(X, y, p.num_cycles, p.learn_rate, cfg.cv_folds);
    params = [
        optimizableVariable('num_cycles', [10,  500], 'Type', 'integer')
        optimizableVariable('learn_rate', [0.01, 1],  'Transform', 'log')
    ];
    results = bayesopt(obj_fn, params, ...
        'MaxObjectiveEvaluations', cfg.tuning_iterations, 'Verbose', 0);
    best_params.num_cycles = results.XAtMinObjective.num_cycles;
    best_params.learn_rate = results.XAtMinObjective.learn_rate;
    best_model = fitrensemble(X, y, 'Method', 'LSBoost', ...
        'NumLearningCycles', best_params.num_cycles, ...
        'LearnRate',         best_params.learn_rate);
    history = results;
end


% -------------------------------------------------------------------------
% Cross-validation error functions
% -------------------------------------------------------------------------

function rmse = cv_error_ridge(X, y, lambda, folds)
    cv   = cvpartition(length(y), 'KFold', folds);
    errs = zeros(folds, 1);
    for f = 1:folds
        tr = training(cv, f); te = test(cv, f);
        m  = fitrlinear(X(tr,:), y(tr), 'Learner', 'leastsquares', ...
            'Regularization', 'ridge', 'Lambda', lambda);
        errs(f) = sqrt(mean((y(te) - predict(m, X(te,:))).^2));
    end
    rmse = mean(errs);
end


function rmse = cv_error_lasso(X, y, lambda, folds)
    cv   = cvpartition(length(y), 'KFold', folds);
    errs = zeros(folds, 1);
    for f = 1:folds
        tr = training(cv, f); te = test(cv, f);
        [B, fi] = lasso(X(tr,:), y(tr), 'Lambda', lambda);
        if isempty(B)
            errs(f) = inf;
        else
            errs(f) = sqrt(mean((y(te) - X(te,:)*B - fi.Intercept).^2));
        end
    end
    rmse = mean(errs);
end


function rmse = cv_error_svm(X, y, C, sigma, epsilon, folds)
    cv   = cvpartition(length(y), 'KFold', folds);
    errs = zeros(folds, 1);
    for f = 1:folds
        tr = training(cv, f); te = test(cv, f);
        m  = fitrsvm(X(tr,:), y(tr), 'KernelFunction', 'gaussian', ...
            'BoxConstraint', C, 'KernelScale', sigma, 'Epsilon', epsilon, ...
            'Standardize', true);
        errs(f) = sqrt(mean((y(te) - predict(m, X(te,:))).^2));
    end
    rmse = mean(errs);
end


function rmse = cv_error_tree(X, y, min_leaf, max_splits, folds)
    cv   = cvpartition(length(y), 'KFold', folds);
    errs = zeros(folds, 1);
    for f = 1:folds
        tr = training(cv, f); te = test(cv, f);
        m  = fitrtree(X(tr,:), y(tr), 'MinLeafSize', min_leaf, 'MaxNumSplits', max_splits);
        errs(f) = sqrt(mean((y(te) - predict(m, X(te,:))).^2));
    end
    rmse = mean(errs);
end


function rmse = cv_error_rf(X, y, num_trees, min_leaf, folds)
    cv   = cvpartition(length(y), 'KFold', folds);
    errs = zeros(folds, 1);
    for f = 1:folds
        tr = training(cv, f); te = test(cv, f);
        m  = TreeBagger(num_trees, X(tr,:), y(tr), 'Method', 'regression', 'MinLeafSize', min_leaf);
        yp = predict(m, X(te,:));
        if iscell(yp), yp = str2double(yp); end
        errs(f) = sqrt(mean((y(te) - yp).^2));
    end
    rmse = mean(errs);
end


function rmse = cv_error_ensemble(X, y, num_cycles, learn_rate, folds)
    cv   = cvpartition(length(y), 'KFold', folds);
    errs = zeros(folds, 1);
    for f = 1:folds
        tr = training(cv, f); te = test(cv, f);
        m  = fitrensemble(X(tr,:), y(tr), 'Method', 'LSBoost', ...
            'NumLearningCycles', num_cycles, 'LearnRate', learn_rate);
        errs(f) = sqrt(mean((y(te) - predict(m, X(te,:))).^2));
    end
    rmse = mean(errs);
end
