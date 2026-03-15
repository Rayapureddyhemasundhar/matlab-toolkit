% ============================================================================
% UNIT TESTS MODULE
% Comprehensive test suite for all pipeline components
% ============================================================================

function unit_tests()

    addpath(genpath('../src'));
    addpath(genpath('../utils'));

    fprintf('%s\n', repmat('=', 1, 60));
    fprintf('RUNNING UNIT TESTS\n');
    fprintf('%s\n\n', repmat('=', 1, 60));

    total  = 0;
    passed = 0;
    failed = 0;

    tests = {
        'Configuration',         @test_config
        'Safe Normalization',    @test_normalization
        'Missing Value Handler', @test_imputation
        'Outlier Detection',     @test_outlier_detection
        'Feature Selection',     @test_feature_selection
        'Model Training',        @test_model_training
        'Feature Engineering',   @test_feature_engineering
    };

    for i = 1:size(tests, 1)
        name = tests{i, 1};
        fn   = tests{i, 2};
        total = total + 1;

        fprintf('Test %d: %s\n', i, name);
        try
            fn();
            passed = passed + 1;
            fprintf('  PASS\n\n');
        catch ME
            failed = failed + 1;
            fprintf('  FAIL: %s\n\n', ME.message);
        end
    end

    fprintf('%s\n', repmat('=', 1, 60));
    fprintf('TEST SUMMARY\n');
    fprintf('%s\n', repmat('-', 1, 60));
    fprintf('Total:  %d\n', total);
    fprintf('Passed: %d\n', passed);
    fprintf('Failed: %d\n', failed);
    fprintf('%s\n', repmat('=', 1, 60));

    if failed == 0
        fprintf('ALL TESTS PASSED\n');
    else
        fprintf('SOME TESTS FAILED\n');
    end

end


% -------------------------------------------------------------------------
% Individual test functions
% -------------------------------------------------------------------------

function test_config()
    cfg = config();
    assert(isstruct(cfg),                           'Config must be a struct');
    assert(isfield(cfg, 'dataset'),                 'Config missing: dataset');
    assert(isfield(cfg, 'train_ratio'),             'Config missing: train_ratio');
    assert(cfg.train_ratio > 0 && cfg.train_ratio < 1, 'train_ratio must be in (0,1)');
    assert(cfg.random_seed >= 0,                    'random_seed must be non-negative');
end


function test_normalization()
    data  = [1, 2, 3; 4, 5, 6; 7, 8, 9];

    % Standard scaler: zero mean, unit variance per column
    [sc, params] = safe_normalization(data, 'standard');
    assert(all(abs(mean(sc)) < 1e-10),              'Standard: column means not zero');

    % Inverse transform must recover original
    recovered = inverse_normalization(sc, params);
    assert(norm(data - recovered) < 1e-8,           'Standard: inverse transform failed');

    % Constant column must not produce NaN/Inf
    data_const = [1, 5, 1; 2, 5, 2; 3, 5, 3];
    sc_const   = safe_normalization(data_const, 'standard');
    assert(all(isfinite(sc_const(:))),              'Constant column produced non-finite values');

    % MinMax: all values in [0,1]
    [sc_mm, ~] = safe_normalization(data, 'minmax');
    assert(max(sc_mm(:)) <= 1 + 1e-10,             'MinMax: max > 1');
    assert(min(sc_mm(:)) >= 0 - 1e-10,             'MinMax: min < 0');
end


function test_imputation()
    % Linear interpolation
    x      = [1; NaN; 3; 4; NaN; 6];
    filled = fillmissing(x, 'linear');
    assert(abs(filled(2) - 2) < 1e-8,              'Linear interpolation incorrect');

    % Mean imputation
    filled_m = fillmissing_by_mean(x);
    expected = mean(x, 'omitnan');
    assert(abs(filled_m(2) - expected) < 1e-8,     'Mean imputation incorrect');

    % KNN imputation: no NaN in output
    X2d = [1, 2; NaN, 4; 3, 6; 4, 8];
    filled_k = fillmissing_by_knn(X2d, 2);
    assert(~any(isnan(filled_k(:))),                'KNN imputation left NaN values');
end


function test_outlier_detection()
    rng(42);
    data = [randn(100, 1); 15; -15; 18];

    % Z-score should flag the injected outliers
    z        = abs((data - mean(data)) / std(data));
    outliers = z > 3;
    assert(sum(outliers) >= 3,                      'Z-score missed injected outliers');

    % IQR should also flag them
    Q1   = prctile(data, 25);
    Q3   = prctile(data, 75);
    IQR  = Q3 - Q1;
    oiqr = data < Q1 - 1.5*IQR | data > Q3 + 1.5*IQR;
    assert(sum(oiqr) >= 3,                          'IQR missed injected outliers');

    % Full function
    cfg = config();
    data2d = [randn(100,3); 15*ones(3,3)];
    [Xc, res] = detect_outliers_comprehensive(data2d, cfg);
    assert(isfield(res, 'total_outliers'),           'Result missing: total_outliers');
    assert(isfield(res, 'consensus'),                'Result missing: consensus');
    assert(~any(isnan(Xc(:))),                       'Winsorization left NaN values');
end


function test_feature_selection()
    rng(42);
    X = randn(100, 10);
    y = 2*X(:,1) + 0.5*X(:,2) + 0.1*randn(100,1);
    cfg = config();

    % PCA must reduce dimensionality
    [Xp, ~, ip] = feature_selection(X, y, 'pca', cfg);
    assert(size(Xp, 2) <= size(X, 2),               'PCA increased dimensionality');
    assert(ip.n_components > 0,                      'PCA returned 0 components');

    % Correlation filter
    [Xc, ~, ~] = feature_selection(X, y, 'correlation', cfg);
    assert(size(Xc, 2) <= size(X, 2),               'Correlation filter increased features');

    % Variance threshold
    [Xv, ~, ~] = feature_selection(X, y, 'variance', cfg);
    assert(size(Xv, 2) <= size(X, 2),               'Variance filter increased features');
end


function test_model_training()
    rng(42);
    X = randn(100, 5);
    y = 2*X(:,1) + 0.5*X(:,2) + 0.1*randn(100,1);

    cfg = config();
    cfg.hyperparameter_tuning = false;
    cfg.ml_models = {'Linear', 'Ridge', 'Tree'};

    n_tr  = 80;
    X_tr  = X(1:n_tr,:);  y_tr = y(1:n_tr);
    X_te  = X(n_tr+1:end,:); y_te = y(n_tr+1:end);

    results = train_all_models(X_tr, y_tr, X_te, y_te, cfg);

    assert(isstruct(results),                        'train_all_models must return a struct');
    assert(~isempty(results.models),                 'No models were trained');
    assert(min(results.rmse) < 1,                    'All models have suspiciously high RMSE');
    assert(length(results.models) == length(results.rmse), 'Model/RMSE count mismatch');
end


function test_feature_engineering()
    rng(42);
    X    = randn(50, 3);
    orig = {'A', 'B', 'C'};
    cfg  = config();

    [Xe, names] = engineer_features_comprehensive(X, orig, cfg);

    assert(size(Xe, 2) > size(X, 2),                'No new features were created');
    assert(length(names) == size(Xe, 2),             'Feature name count mismatch');
    assert(all(isfinite(Xe(:))),                     'Engineered features contain non-finite values');
end


% -------------------------------------------------------------------------
% Local copies of helper functions (so tests run standalone)
% -------------------------------------------------------------------------

function filled = fillmissing_by_mean(x)
    filled   = x;
    col_mean = mean(x, 'omitnan');
    if isnan(col_mean), col_mean = 0; end
    filled(isnan(x)) = col_mean;
end


function filled = fillmissing_by_knn(X, k)
    filled = X;
    for col = 1:size(X, 2)
        missing = isnan(X(:, col));
        if ~any(missing), continue; end
        other  = setdiff(1:size(X,2), col);
        comp   = ~any(isnan(X(:, other)), 2) & ~missing;
        if sum(comp) > k
            Xtr = X(comp, other);
            ytr = X(comp, col);
            for r = find(missing)'
                if ~any(isnan(X(r, other)))
                    d = sqrt(sum((Xtr - X(r, other)).^2, 2));
                    [~, idx] = mink(d, min(k, length(d)));
                    filled(r, col) = mean(ytr(idx));
                else
                    filled(r, col) = mean(ytr);
                end
            end
        else
            filled(missing, col) = mean(X(~missing, col), 'omitnan');
        end
    end
end
