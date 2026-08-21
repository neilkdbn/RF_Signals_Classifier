% generate_custom_rf_v4.m
% =========================================================================
% RF SIGNAL CLASSIFICATION: PHASE 2 ROBUST SIGNAL GENERATOR (v4)
% Generates simulated, channel-impaired communications waveforms.
% Works out-of-the-box with or without the Communications Toolbox!
% Features an interactive Save Dialog for seamless saving in MATLAB Online.
% =========================================================================

clear; clc; close all;

fprintf('==================================================\n');
fprintf('     MATLAB CUSTOM RF SIGNAL GENERATOR ACTIVATED   \n');
fprintf('==================================================\n\n');

%% 1. Configuration & Parameters
numSamplesPerFrame = 128;       % Segment length matching RadioML2016 structure
numFramesPerMod = 1000;         % Number of frames to generate per modulation scheme
sps = 8;                         % Samples per symbol
fs = 200e3;                      % Sampling rate (200 kHz)

% Modulation schemes to generate (5 Core Classes)
modulations = {'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'};
numClasses = length(modulations);

% Channel Impairments Config
snr_dB = 10;                     % Target Signal-to-Noise Ratio (dB)
cfo_Hz = 150;                    % Center Frequency Offset (Hz)

fprintf('Configuring signal parameters:\n');
fprintf('-> Base Sample Rate: %.1f kHz\n', fs/1e3);
fprintf('-> Samples Per Frame: %d (I/Q complex vectors)\n', numSamplesPerFrame);
fprintf('-> Target Signal SNR: %d dB\n\n', snr_dB);

% Check for Communications Toolbox
hasCommToolbox = license('test', 'Communication_Toolbox') && ~isempty(ver('comm'));
if hasCommToolbox
    fprintf('[INFO] Communications Toolbox detected! Using high-fidelity channel models.\n\n');
else
    fprintf('[WARNING] Communications Toolbox not found. Using robust mathematical fallbacks.\n\n');
end

%% 2. Setup Channel Objects (MathWorks Communications Toolbox)
ricianChan = [];
if hasCommToolbox
    % Setup Rician fading object OUTSIDE the loop to avoid persistent script errors
    ricianChan = comm.RicianChannel(...
        'SampleRate', fs, ...
        'KFactor', 4, ...
        'PathDelays', [0 1.8 3.4]/fs, ...
        'AveragePathGains', [0 -3 -5], ...
        'MaximumDopplerShift', 5, ...
        'Visualization', 'Off');
end

%% 3. Initialize Arrays
totalFrames = numFramesPerMod * numClasses;
X_custom = zeros(totalFrames, 2, numSamplesPerFrame);
y_custom = zeros(totalFrames, 1); % Class label (0 to numClasses-1)

frameIdx = 1;

%% 4. Signal Generation Loop
for mIdx = 1:numClasses
    modName = modulations{mIdx};
    fprintf('Generating %s signals (%d frames)... \n', modName, numFramesPerMod);
    
    for fIdx = 1:numFramesPerMod
        numSymbols = numSamplesPerFrame / sps;
        
        % Step A: Generate random symbols and modulate
        if hasCommToolbox
            switch modName
                case 'BPSK'
                    symbols = randi([0 1], numSymbols, 1);
                    modulatedSig = pskmod(symbols, 2);
                case 'QPSK'
                    symbols = randi([0 3], numSymbols, 1);
                    modulatedSig = pskmod(symbols, 4, pi/4);
                case '8PSK'
                    symbols = randi([0 7], numSymbols, 1);
                    modulatedSig = pskmod(symbols, 8);
                case '16QAM'
                    symbols = randi([0 15], numSymbols, 1);
                    modulatedSig = qammod(symbols, 16, 'UnitAveragePower', true);
                case '64QAM'
                    symbols = randi([0 63], numSymbols, 1);
                    modulatedSig = qammod(symbols, 64, 'UnitAveragePower', true);
            end
        else
            % Mathematical fallback: Manual Modulation Mapping
            switch modName
                case 'BPSK'
                    symbols = randi([0 1], numSymbols, 1);
                    modulatedSig = 2*symbols - 1; % BPSK mapping
                case 'QPSK'
                    symbols = randi([0 3], numSymbols, 1);
                    modulatedSig = exp(1i * (symbols * pi/2 + pi/4)); % QPSK mapping
                case '8PSK'
                    symbols = randi([0 7], numSymbols, 1);
                    modulatedSig = exp(1i * symbols * pi/4);
                case '16QAM'
                    symbols = randi([0 15], numSymbols, 1);
                    qam_real = 2*mod(symbols, 4) - 3;
                    qam_imag = 3 - 2*floor(symbols / 4);
                    modulatedSig = (qam_real + 1i*qam_imag) / sqrt(10); % Normalized power
                case '64QAM'
                    symbols = randi([0 63], numSymbols, 1);
                    qam_real = 2*mod(symbols, 8) - 7;
                    qam_imag = 7 - 2*floor(symbols / 8);
                    modulatedSig = (qam_real + 1i*qam_imag) / sqrt(42); % Normalized power
            end
        end
        
        % Step B: Upsample to simulate samples-per-symbol (sps)
        txSig = zeros(numSamplesPerFrame, 1);
        txSig(1:sps:end) = modulatedSig;
        
        % Step C: Pulse Shaping / Filtering
        % Simple raised cosine filter approximation
        t_filt = -2:1/sps:2;
        rrc_filter = sinc(t_filt) .* cos(pi*0.35*t_filt) ./ (1 - (2*0.35*t_filt).^2 + eps);
        rrc_filter = rrc_filter / norm(rrc_filter);
        txSig = conv(txSig, rrc_filter, 'same');
        txSig = txSig(1:numSamplesPerFrame);
        
        % Step D: Channel fading (Multipath & Doppler)
        if hasCommToolbox
            fadedSig = ricianChan(txSig);
        else
            % Mathematical fallback: Simple multi-path fading simulator
            pathGains = [1.0, 0.7*exp(1i*pi/3), 0.5*exp(-1i*pi/4)];
            fadedSig = txSig * pathGains(1) + ...
                       [0; txSig(1:end-1)] * pathGains(2) + ...
                       [0; 0; txSig(1:end-2)] * pathGains(3);
        end
        
        % Step E: Apply Carrier Frequency Offset (CFO)
        t = (0:numSamplesPerFrame-1)' / fs;
        cfoSig = fadedSig .* exp(1i * 2 * pi * cfo_Hz * t);
        
        % Step F: Add Noise (AWGN)
        sigPower = mean(abs(cfoSig).^2);
        noisePower = sigPower / (10^(snr_dB/10));
        noise = sqrt(noisePower/2) * (randn(size(cfoSig)) + 1i*randn(size(cfoSig)));
        rxSig = cfoSig + noise;
        
        % Step G: Normalise (Zero-mean, Unit-variance)
        rxSig = rxSig - mean(rxSig);
        rxSig = rxSig / (std(rxSig) + 1e-6);
        
        % Step H: Store real (In-phase) and imag (Quadrature) components
        X_custom(frameIdx, 1, :) = real(rxSig);
        X_custom(frameIdx, 2, :) = imag(rxSig);
        y_custom(frameIdx) = mIdx - 1; % 0-indexed for Python
        
        frameIdx = frameIdx + 1;
    end
end

%% 5. Save Dataset to Disk (With MATLAB Online Interactive Save Dialog)
outputFile = 'custom_rf_dataset.mat';

% Force MATLAB Online to open a save-file dialog window.
% This allows you to explicitly click on and save into your desired MATLAB Drive folders.
try
    fprintf('\n==================================================\n');
    fprintf('     SELECT YOUR MATLAB DRIVE FOLDER TO SAVE      \n');
    fprintf('==================================================\n');
    fprintf('Opening save dialog... Please select your MATLAB Drive folder to save the dataset.\n\n');
    
    [filename, pathname] = uiputfile('custom_rf_dataset.mat', 'Save Custom Dataset to MATLAB Drive');
    
    if isequal(filename,0) || isequal(pathname,0)
        % User cancelled the dialog, fallback to saving in current active workspace folder
        fprintf('[WARNING] Save dialog was closed or canceled.\n');
        save(outputFile, 'X_custom', 'y_custom', 'modulations', 'fs', 'snr_dB');
        fprintf('-> Saved to default workspace path: %s\n', fullfile(pwd, outputFile));
    else
        % Save directly to the user's selected MATLAB Drive folder
        save(fullfile(pathname, filename), 'X_custom', 'y_custom', 'modulations', 'fs', 'snr_dB');
        fprintf('-> SUCCESS: Saved to your chosen folder: %s\n', fullfile(pathname, filename));
    end
catch ME
    % Fallback if UI dialog fails (e.g. running headlessly or non-interactively)
    save(outputFile, 'X_custom', 'y_custom', 'modulations', 'fs', 'snr_dB');
    fprintf('[INFO] Save dialog bypassed. Saved to default path: %s\n', fullfile(pwd, outputFile));
end

%% 6. Plot a preview of the last generated frame (64-QAM)
try
    figure('Color', 'w');
    subplot(2,1,1);
    plot(1:numSamplesPerFrame, squeeze(X_custom(end, 1, :)), 'b', 'LineWidth', 1.5); hold on;
    plot(1:numSamplesPerFrame, squeeze(X_custom(end, 2, :)), 'r', 'LineWidth', 1.5);
    title(['Time-Domain Signal: ' modName ' at ' num2str(snr_dB) ' dB SNR']);
    xlabel('Sample Index'); ylabel('Amplitude');
    legend('In-Phase (I)', 'Quadrature (Q)');
    grid on;

    subplot(2,1,2);
    plot(squeeze(X_custom(end, 1, :)), squeeze(X_custom(end, 2, :)), 'o', ...
        'MarkerFaceColor', '#3498db', 'MarkerEdgeColor', 'b');
    title(['Constellation Diagram of received ' modName]);
    xlabel('In-Phase (I)'); ylabel('Quadrature (Q)');
    axis square; grid on;
catch
    % Bypassed if graphics system is constrained in cloud container
end

fprintf('\n==================================================\n');
fprintf('     SUCCESS: CUSTOM SIGNAL GENERATION COMPLETE    \n');
fprintf('     Generated %d total samples across %d classes  \n', totalFrames, numClasses);
fprintf('==================================================\n');
