function [S, t, f] = eventRelatedSpectrogram(signal, fs, eventTimes, varargin)
% eventRelatedSpectrogram - Compute event-related time-frequency spectrogram
%
% Computes a spectrogram time-locked to events by extracting signal epochs
% around each event time, computing the power spectral density using the
% short-time Fourier transform, and averaging across epochs.
%
% Usage:
%   [S, t, f] = eventRelatedSpectrogram(signal, fs, eventTimes)
%   [S, t, f] = eventRelatedSpectrogram(signal, fs, eventTimes, Name, Value)
%
% Inputs:
%   signal      - 1-D signal vector (samples x 1 or 1 x samples)
%   fs          - Sampling frequency in Hz
%   eventTimes  - Vector of event times in seconds
%
% Optional Name-Value Pairs:
%   'Window'        - Analysis window duration in seconds (default: 0.25)
%   'PreEvent'      - Time before event in seconds (default: 1)
%   'PostEvent'     - Time after event in seconds (default: 2)
%   'Overlap'       - Window overlap fraction [0, 1) (default: 0.5)
%   'FreqRange'     - [fMin fMax] frequency range in Hz (default: [0 fs/2])
%   'Baseline'      - [tStart tEnd] baseline window in seconds relative
%                     to event (default: [] = no normalization)
%   'BaselineType'  - Baseline normalization type: 'zscore', 'dB', or
%                     'percent' (default: 'dB')
%   'Plot'          - Plot result (default: false)
%
% Outputs:
%   S  - Averaged power spectrogram (frequencies x time bins)
%   t  - Time vector relative to event (seconds)
%   f  - Frequency vector (Hz)
%
% Example:
%   fs = 1000;
%   t  = 0:1/fs:10;
%   signal = randn(size(t));
%   eventTimes = [1, 3, 5, 7];
%   [S, t_ax, f_ax] = eventRelatedSpectrogram(signal, fs, eventTimes, ...
%       'PreEvent', 0.5, 'PostEvent', 1, 'Baseline', [-0.5, 0], 'Plot', true);

% Parse inputs
p = inputParser;
addRequired(p, 'signal',     @(x) isnumeric(x) && isvector(x));
addRequired(p, 'fs',         @(x) isnumeric(x) && isscalar(x) && x > 0);
addRequired(p, 'eventTimes', @(x) isnumeric(x) && isvector(x));
addParameter(p, 'Window',       0.25,   @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'PreEvent',     1,      @(x) isnumeric(x) && isscalar(x) && x >= 0);
addParameter(p, 'PostEvent',    2,      @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'Overlap',      0.5,    @(x) isnumeric(x) && isscalar(x) && x >= 0 && x < 1);
addParameter(p, 'FreqRange',    [],     @(x) isempty(x) || (isnumeric(x) && numel(x) == 2 && x(1) < x(2)));
addParameter(p, 'Baseline',     [],     @(x) isempty(x) || (isnumeric(x) && numel(x) == 2 && x(1) < x(2)));
addParameter(p, 'BaselineType', 'dB',   @(x) ismember(x, {'zscore', 'dB', 'percent'}));
addParameter(p, 'Plot',         false,  @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
parse(p, signal, fs, eventTimes, varargin{:});

opts   = p.Results;
signal = signal(:);  % ensure column vector

% STFT parameters in samples
winSamples = round(opts.Window * fs);
noverlap   = round(opts.Overlap * winSamples);
epochPre   = round(opts.PreEvent  * fs);
epochPost  = round(opts.PostEvent * fs);
nSamples   = length(signal);

% Get frequency and time axes from a dummy epoch
[~, f, tRel] = spectrogram(zeros(epochPre + epochPost + 1, 1), winSamples, noverlap, [], fs);
tRel = tRel - opts.PreEvent;  % shift so t=0 is the event

% Apply frequency range mask
if isempty(opts.FreqRange)
    fMask = true(size(f));
else
    fMask = f >= opts.FreqRange(1) & f <= opts.FreqRange(2);
end
f = f(fMask);

% Accumulate power spectrograms across epochs
nEvents = numel(eventTimes);
Ssum    = zeros(sum(fMask), numel(tRel));
nValid  = 0;

for k = 1:nEvents
    idx0   = round(eventTimes(k) * fs);  % event sample (1-based)
    iStart = idx0 - epochPre + 1;
    iEnd   = idx0 + epochPost + 1;

    if iStart < 1 || iEnd > nSamples
        warning('eventRelatedSpectrogram:epochOutOfBounds', ...
            'Event %d (t=%.3f s) epoch extends beyond signal bounds; skipping.', ...
            k, eventTimes(k));
        continue;
    end

    epoch  = signal(iStart:iEnd);
    Sk     = spectrogram(epoch, winSamples, noverlap, [], fs);
    Ssum   = Ssum + abs(Sk(fMask, :)).^2;
    nValid = nValid + 1;
end

if nValid == 0
    error('eventRelatedSpectrogram:noValidEpochs', ...
        'No valid epochs found. Check that event times fall within the signal.');
end

S = Ssum / nValid;

% Baseline normalization
if ~isempty(opts.Baseline)
    bMask = tRel >= opts.Baseline(1) & tRel <= opts.Baseline(2);
    if ~any(bMask)
        warning('eventRelatedSpectrogram:baselineEmpty', ...
            'No time bins fall within the specified baseline window; skipping normalization.');
    else
        baseline = mean(S(:, bMask), 2);  % mean power per frequency bin
        switch opts.BaselineType
            case 'dB'
                S = 10 * log10(bsxfun(@rdivide, S, baseline));
            case 'zscore'
                baseStd        = std(S(:, bMask), 0, 2);
                baseStd(baseStd == 0) = 1;
                S = bsxfun(@rdivide, bsxfun(@minus, S, baseline), baseStd);
            case 'percent'
                S = 100 * bsxfun(@rdivide, bsxfun(@minus, S, baseline), baseline);
        end
    end
end

t = tRel;

% Optional plot
if opts.Plot
    figure;
    imagesc(t, f, S);
    axis xy;
    xlabel('Time relative to event (s)');
    ylabel('Frequency (Hz)');
    if ~isempty(opts.Baseline)
        switch opts.BaselineType
            case 'dB',      cbLabel = 'Power (dB)';
            case 'zscore',  cbLabel = 'Power (z-score)';
            case 'percent', cbLabel = 'Power change (%)';
        end
    else
        cbLabel = 'Power';
    end
    colorbar('Label', cbLabel);
    title(sprintf('Event-Related Spectrogram (n=%d events)', nValid));
    xline(0, 'w--', 'LineWidth', 1.5);
end

end
