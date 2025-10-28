function trab_prat_d()


clear; clc; close all;


classes = {'circle', 'kite', 'parallelogram', 'square', 'trapezoid', 'triangle'};
numClasses = numel(classes);
imgsPerClass = 5;
imgSize = [32, 32];
nomeRedes = {'rede_c)iii)_1.mat', 'rede_c)iii)_2.mat', 'rede_c)iii)_3.mat'};


inputData = [];
targetData = [];

for i = 1:numClasses
    folder = fullfile('alinea_d', classes{i});
    files = [dir(fullfile(folder, '*.png')); dir(fullfile(folder, '*.jpg'))];

    for j = 1:min(imgsPerClass, length(files))
        img = imread(fullfile(folder, files(j).name));
        if size(img,3) == 3
            img = rgb2gray(img);
        end
        img = imbinarize(imresize(img, imgSize));
        img = ~img;

        inputData(:, end+1) = img(:);
        t = zeros(numClasses,1); t(i) = 1;
        targetData(:, end+1) = t;
    end
end


for r = 1:length(nomeRedes)
    fprintf('\n\n---REDE: %s---\n', nomeRedes{r});

    % Carregar rede
    dados = load(nomeRedes{r});
    nomes = fieldnames(dados);
    net = dados.(nomes{1});

    % Classificar
    out = net(inputData);

    % Cálculo da precisão global (manual)
    acertosTotal = 0;
    for i = 1:size(out,2)
        [~, pred] = max(out(:,i));
        [~, real] = max(targetData(:,i));
        if pred == real
            acertosTotal = acertosTotal + 1;
        end
    end
    accTotal = acertosTotal / size(out,2) * 100;
    fprintf('Precisão GLOBAL (total): %.2f%% (%d/%d)\n\n', accTotal, acertosTotal, size(out,2));

    % Cálculo da precisão por classe (manual)
    fprintf('Precisão por classe:\n');
    for c = 1:numClasses
        totalClasse = 0;
        acertosClasse = 0;
        for i = 1:size(out,2)
            [~, pred] = max(out(:,i));
            [~, real] = max(targetData(:,i));
            if real == c
                totalClasse = totalClasse + 1;
                if pred == real
                    acertosClasse = acertosClasse + 1;
                end
            end
        end
        accClasse = acertosClasse / totalClasse * 100;
        fprintf('  %-12s : %.2f%% (%d/%d)\n', classes{c}, accClasse, acertosClasse, totalClasse);
    end

    % Matriz de confusão
    figure;
    plotconfusion(targetData, out);
    title(['Matriz de Confusão - ', nomeRedes{r}], 'Interpreter', 'none');
end
end
