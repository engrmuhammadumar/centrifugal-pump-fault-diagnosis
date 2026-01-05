clc; clear; close all;

%% ===================== SETTINGS =====================
fs = 25600;   % Sampling frequency (25.6 kHz)

% INPUT and OUTPUT paths
inRoot   = 'E:\CP Dataset\[20200428] PUMP DATA\Vibration\4 bar';
saveRoot = 'F:\CP Data\4 bar';

datasets = {
    'Impeller crack(4.0bar)'
    'Mechanical Seal Hole(4.0bar)'
    'Mechanical Seal Scratch(4.0bar)'
    'Normal(4.0bar)'
};

%% ===================== LOAD BASELINE =====================
baselinePath = 'E:\CP Dataset\h_channel1_baseline.mat';
load(baselinePath);                 % variable: h_channel1_baseline

baseline = h_channel1_baseline(:);
baseline = baseline - mean(baseline);

fprintf('Baseline loaded: %d samples\n', length(baseline));

%% ===================== MAIN LOOP =====================
for d = 1:numel(datasets)

    folderName = datasets{d};

    inFolder  = fullfile(inRoot, folderName);
    outFolder = fullfile(saveRoot, folderName);

    if ~exist(outFolder,'dir')
        mkdir(outFolder);
    end

    files = dir(fullfile(inFolder,'*.mat'));
    fprintf('[%s] Found %d files\n', folderName, numel(files));

    for k = 1:numel(files)

        filePath = fullfile(files(k).folder, files(k).name);
        data = load(filePath);

        % ---------- Extract signal (channel 3) ----------
        sig = data.signal(3,:);
        sig = sig(:) - mean(sig);

        % ---------- Match baseline & signal lengths ----------
        N = min(length(baseline), length(sig));
        x = baseline(1:N);
        y = sig(1:N);

        % ---------- Wavelet Coherence ----------
        [Wcoh,~,f,t] = wcoherence(x, y, fs);

        % ---------- Plot RAW WCA (image only) ----------
        fig = figure('Visible','off','Color','w');
        imagesc(t, f, abs(Wcoh));
        axis xy;
        axis off;
        colormap(jet);

        % ---------- Save IMAGE ----------
        outName = fullfile(outFolder, sprintf('WCA_%04d.png', k));
        exportgraphics(gca, outName, 'Resolution', 300);

        close(fig);
    end
end

disp('DONE: WCA images generated successfully');
