function trab_prat_c()


clear; clc; close all;


classes = {'circle', 'kite', 'parallelogram', 'square', 'trapezoid', 'triangle'};
numClasses = numel(classes);
imgSize = [32, 32];
redes_b = {'rede_com_96_global_83_teste.mat',...
    'rede_com_97_global_73_teste.mat',...
    'rede_com_98_global_73_teste.mat'};


melhoresRedes = struct(); % guarda o desempenho geral

% Função auxiliar para carregar imagens de uma pasta
loadImages = @(folderBase) carregarImagens(folderBase, classes, imgSize);

% c.i) Testar redes guardadas de b) com imagens de 'test'
fprintf('\n--- c.i) Avaliar redes de b) com imagens da pasta test ---\n');
[inputTest, targetTest] = loadImages('test');
for i = 1:length(redes_b)
    nomeRede = redes_b{i};
    fprintf('\n Rede %s\n', nomeRede);
    load(nomeRede, 'net');

    out = net(inputTest);
    [acc,~] = avaliarRede(targetTest, out);
    fprintf('Precisão em test: %.2f%%\n', acc);

    figure;
    plotconfusion(targetTest, out);
    title(['Matriz de Confusão - ', nomeRede]);

    melhoresRedes(i).nome = nomeRede;
    melhoresRedes(i).acc_ci = acc;
end

% c.ii) Treinar redes com imagens de 'test' e avaliar nas 3 pastas
fprintf('\n--- c.ii) Treinar redes com imagens da pasta test ---\n');
[inputsTest, targetsTest] = loadImages('test');
[inputsStart, targetsStart] = loadImages('start');
[inputsTrain, targetsTrain] = loadImages('train');

for i = 1:length(redes_b)
    fprintf('\n Treinar nova rede com dados de test...\n');
    net = feedforwardnet(10, 'trainlm');
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'purelin';
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    net.trainParam.epochs = 100;

    [net, ~] = train(net, inputsTest, targetsTest);

    % Testar nos 3 conjuntos
    outStart = net(inputsStart); [accS, ~] = avaliarRede(targetsStart, outStart);
    outTrain = net(inputsTrain); [accT, ~] = avaliarRede(targetsTrain, outTrain);
    outTest  = net(inputsTest);  [accTe, ~] = avaliarRede(targetsTest, outTest);

    fprintf('Precisões: start=%.2f%% | train=%.2f%% | test=%.2f%%\n', accS, accT, accTe);

    figure;
    plotconfusion(targetsStart, outStart);
    title(sprintf('Matriz de Confusão - Rede %d - Pasta start', i), 'Interpreter', 'none');
    
    figure;
    plotconfusion(targetsTrain, outTrain);
    title(sprintf('Matriz de Confusão - Rede %d - Pasta train', i), 'Interpreter', 'none');
    
    figure;
    plotconfusion(targetsTest, outTest);
    title(sprintf('Matriz de Confusão - Rede %d - Pasta test', i), 'Interpreter', 'none');

    melhoresRedes(i).acc_cii_start = accS;
    melhoresRedes(i).acc_cii_train = accT;
    melhoresRedes(i).acc_cii_test  = accTe;
end


% c.iii) Treinar redes com start + train + test e avaliar nas 3
fprintf('\n--- c.iii) Treinar com TODAS as imagens ---\n');
[inputsAll, targetsAll] = loadImagesAll({'start','train','test'}, classes, imgSize);

for i = 1:length(redes_b)
    fprintf('\n Treinar rede completa com todos os dados...\n');
    net = feedforwardnet(10, 'trainlm');
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'purelin';
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;
    net.trainParam.epochs = 100;

    [net, ~] = train(net, inputsAll, targetsAll);

    % Avaliar
    outStart = net(inputsStart); [accS, ~] = avaliarRede(targetsStart, outStart);
    outTrain = net(inputsTrain); [accT, ~] = avaliarRede(targetsTrain, outTrain);
    outTest  = net(inputsTest);  [accTe, ~] = avaliarRede(targetsTest, outTest);

    fprintf('Todas: start=%.2f%% | train=%.2f%% | test=%.2f%%\n', accS, accT, accTe);

    figure;
    plotconfusion(targetsStart, outStart);
    title(sprintf('Matriz de Confusão - Rede %d - Pasta start', i), 'Interpreter', 'none');
    
    figure;
    plotconfusion(targetsTrain, outTrain);
    title(sprintf('Matriz de Confusão - Rede %d - Pasta train', i), 'Interpreter', 'none');
    
    figure;
    plotconfusion(targetsTest, outTest);
    title(sprintf('Matriz de Confusão - Rede %d - Pasta test', i), 'Interpreter', 'none');

    melhoresRedes(i).acc_ciii_start = accS;
    melhoresRedes(i).acc_ciii_train = accT;
    melhoresRedes(i).acc_ciii_test  = accTe;

    nomeSave = sprintf('rede_c)iii)_%d.mat', i);
    save(nomeSave, 'net');
end

% c.iv) Guardar as 3 melhores redes (já guardadas acima)
fprintf('\n--- c.iv) As 3 melhores redes foram treinadas e guardadas com prefixo rede_c)iii)_*.mat ---\n');

end

% Função auxiliar: carregar imagens de 1 pasta
function [inputData, targetData] = carregarImagens(folderBase, classes, imgSize)
    inputData = []; targetData = [];
    numClasses = length(classes);
    
    for i = 1:numClasses
        folder = fullfile(folderBase, classes{i});
        files = [dir(fullfile(folder, '*.png')); dir(fullfile(folder, '*.jpg'))];
    
        for j = 1:length(files)
            img = imread(fullfile(folder, files(j).name));
            if size(img,3) == 3, img = rgb2gray(img); end
            img = imbinarize(imresize(img, imgSize));
            inputData(:, end+1) = img(:);
    
            t = zeros(numClasses,1); t(i) = 1;
            targetData(:, end+1) = t;
        end
    end
end

% Função auxiliar: carregar imagens de várias pastas
function [inputs, targets] = loadImagesAll(pastas, classes, imgSize)
    inputs = []; targets = [];
    for p = 1:length(pastas)
        [iData, tData] = carregarImagens(pastas{p}, classes, imgSize);
        inputs = [inputs, iData];
        targets = [targets, tData];
    end
end

% Função auxiliar: avaliar rede
function [acc, cm] = avaliarRede(targets, outputs)
    [~, pred] = max(outputs);
    [~, real] = max(targets);
    acertos = sum(pred == real);
    acc = 100 * acertos / length(real);
    cm = confusionmat(real, pred);
end
