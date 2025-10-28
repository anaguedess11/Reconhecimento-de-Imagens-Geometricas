function trab_prat_b()

clear; clc; close all;


classes = {'circle', 'kite', 'parallelogram', 'square', 'trapezoid', 'triangle'};
numClasses = numel(classes);
imgsPerClass = 50;
imgSize = [32, 32];
inputData = [];
targetData = [];

% Ler e converter imagens
for i = 1:numClasses
    folder = fullfile('train', classes{i});
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

net.divideFcn = 'dividerand';                % Divide os exemplos aleatoriamente
net.divideParam.trainRatio = 0.7;            % treino
net.divideParam.valRatio = 0.15;             % validação
net.divideParam.testRatio = 0.15;            % teste

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


% Calcular precisão de teste

testInputs = inputData(:, tr.testInd);
testTargets = targetData(:, tr.testInd);
outTest = net(testInputs);

acertosTeste = 0;
for i = 1:size(outTest,2)
    [~, pred] = max(outTest(:,i));
    [~, real] = max(testTargets(:,i));
    if pred == real
        acertosTeste = acertosTeste + 1;
    end
end
accTeste = acertosTeste / size(outTest,2) * 100;
fprintf('Precisão TESTE: %.0f%%\n', accTeste);

if accTotal > 93
    nomeFicheiro = sprintf('rede_com_%d_global_%d_teste.mat', round(accTotal), round(accTeste));
    save(nomeFicheiro, 'net');
end

end