% ============================================================================
% LOGGING MODULE
% Professional logging system with timestamps, levels, and file output
% ============================================================================

function varargout = logger(message, level, varargin)
    % Log a message with timestamp and severity level.
    %
    % Usage:
    %   logger('Starting pipeline', 'INFO')
    %   logger('Error occurred',    'ERROR')
    %   logger(repmat('=',1,60),    'INFO')   % separator lines

    persistent log_file log_levels cfg initialized

    if nargin == 0
        if ~isempty(log_file) && log_file ~= -1
            fclose(log_file);
            fprintf('Log closed.\n');
        end
        return;
    end

    if isempty(initialized)
        if exist('config.m', 'file')
            cfg = config();
        else
            cfg.log_level   = 'INFO';
            cfg.log_to_file = true;
            cfg.log_file    = 'logs/pipeline.log';
        end

        log_levels = {'DEBUG', 'INFO', 'WARN', 'ERROR'};

        if cfg.log_to_file
            if ~exist('logs', 'dir')
                mkdir('logs');
            end
            log_file = fopen(cfg.log_file, 'a');
            if log_file == -1
                warning('Unable to open log file for writing: %s. Logging to file disabled.', cfg.log_file);
                cfg.log_to_file = false;
            end
        end

        initialized = true;
    end

    if nargin < 2 || isempty(level)
        level = 'INFO';
    end

    % Filter by log level
    current_level_idx = find(strcmp(log_levels, cfg.log_level), 1);
    if isempty(current_level_idx)
        current_level_idx = find(strcmp(log_levels, 'INFO'), 1);
    end

    msg_level_idx     = find(strcmp(log_levels, level), 1);
    if isempty(msg_level_idx)
        msg_level_idx = current_level_idx;
    end

    if msg_level_idx < current_level_idx
        if nargout > 0
            varargout{1} = '';
        end
        return;
    end

    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');

    if nargin >= 3 && ~isempty(varargin)
        formatted_msg = sprintf(message, varargin{:});
    else
        formatted_msg = message;
    end

    log_entry = sprintf('[%s] [%-5s] %s\n', timestamp, level, formatted_msg);

    % Console output
    switch level
        case {'ERROR', 'WARN'}
            fprintf(2, '%s', log_entry);
        otherwise
            fprintf('%s', log_entry);
    end

    % File output
    if cfg.log_to_file && ~isempty(log_file) && log_file ~= -1
        fprintf(log_file, '%s', log_entry);
        % MATLAB flushes file I/O automatically; fflush is not a built-in function.
    end

    if nargout > 0
        varargout{1} = log_entry;
    end

end
