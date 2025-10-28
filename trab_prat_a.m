function trab_prat_a()


clear; clc; close all;


classes = {'circle', 'kite', 'parallelogram', 'square', 'trapezoid', 'triangle'};
numClasses = numel(classes);
imgsPerClass = 5;
imgSize = [32, 32];
inputData = [];
targetData = [];

% Ler e converter imagens
for i = 1:numClasses
    folder = fullfile('start', classes{i});
    files = [dir(fullfile(folder, '*.png')); dir(fullfile(folder, '*.jpg'))];

    for j = 1:min(imgsPerClass, length(files))
        img = imread(fullfile(folder, files(j).name));
        if size(img,3) == 3
            img = rgb2gray(img);
        end
        img = imbinarize(imresize(img, imgSize));
        inputData(:, end+1) = img(:);
        t = zeros(numClasses,1); t(i) = 1;
        targetData(:, end+1) = t;
    end
end


% Configurações da rede
net = feedforwardnet(10, 'trainlm');         % x camada escondida com y neurónios
net.layers{1}.transferFcn = 'tansig';        % Função de ativação da camada escondida
net.layers{2}.transferFcn = 'purelin';       % Função de ativação da camada de saída

net.divideFcn = '';                          

net.trainParam.epochs = 100;                 % Número máximo de épocas

% Treinar
[net, tr] = train(net, inputData, targetData);

% Precisão global
outAll = net(inputData);

figure; plotconfusion(targetData, outAll);
title('Matriz de Confusão - Total');

acertosTotal = 0;
for i = 1:size(outAll,2)
    [~, pred] = max(outAll(:,i));
    [~, real] = max(targetData(:,i));
    if pred == real
        acertosTotal = acertosTotal + 1;
    end
end
accTotal = acertosTotal / size(outAll,2) * 100;
fprintf('Precisão GLOBAL (total): %.0f%%\n', accTotal);



end