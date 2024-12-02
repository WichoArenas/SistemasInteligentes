% Limpieza del entorno
clear; clc; close all;

% --- Parámetros comunes ---
numDimensions = 2; % Número de dimensiones del problema
xMin = -5; xMax = 5; % Límites del espacio de búsqueda
maxIterations = 50; % Máximo número de iteraciones

% --- Definición de la función objetivo (Rosenbrock) ---
FitnessFunction = @(x) sum(100 * (x(2:end) - x(1:end-1).^2).^2 + (1 - x(1:end-1)).^2);

% Crear una malla para graficar la función objetivo
[x, y] = meshgrid(xMin:0.5:xMax, xMin:0.5:xMax); % Malla de puntos en los ejes X e Y
z = arrayfun(@(x, y) FitnessFunction([x, y]), x, y); % Evaluar la función objetivo en la malla

%% --- PSO: Configuración y Ejecución ---
numParticles = 30; % Número de partículas
w = 0.7;  % Factor de inercia (controla la velocidad previa)
c1 = 1.5; % Coeficiente cognitivo (atracción hacia el mejor personal)
c2 = 1.5; % Coeficiente social (atracción hacia el mejor global)

% Inicialización de PSO
positions = xMin + (xMax - xMin) * rand(numParticles, numDimensions); % Posiciones iniciales aleatorias
velocities = zeros(numParticles, numDimensions); % Velocidades iniciales en cero
pBest = positions; % Mejor posición personal (inicialmente la posición actual)
pBestScores = arrayfun(@(i) FitnessFunction(positions(i,:)), 1:numParticles)'; % Evaluar la función objetivo en cada partícula
[gBestScore, gBestIdx] = min(pBestScores); % Determinar el mejor puntaje global y su índice
gBest = positions(gBestIdx, :); % Mejor posición global

% Configuración de la gráfica para PSO
figure; % Crear una nueva figura
subplot(1, 2, 1); % Subgráfica para PSO
surf(x, y, z, 'EdgeColor', 'none', 'FaceAlpha', 0.7); % Graficar la superficie de la función
hold on;
scatter3(positions(:, 1), positions(:, 2), ...
    arrayfun(@(i) FitnessFunction(positions(i,:)), 1:numParticles)', ...
    50, 'r', 'filled'); % Graficar las partículas
title('PSO - Movimiento de Partículas');
xlabel('x'); ylabel('y'); zlabel('f(x, y)');
grid on;

%% --- GA: Configuración y Ejecución ---
numPopulation = 30; % Tamaño de la población
population = xMin + (xMax - xMin) * rand(numPopulation, numDimensions); % Población inicial aleatoria
gaBestScores = zeros(1, maxIterations); % Arreglo para almacenar los mejores puntajes en cada iteración
mutationRate = 0.1; % Tasa de mutación

% Configuración de la gráfica para GA
subplot(1, 2, 2); % Subgráfica para GA
surf(x, y, z, 'EdgeColor', 'none', 'FaceAlpha', 0.7); % Graficar la superficie de la función
hold on;
scatter3(population(:, 1), population(:, 2), ...
    arrayfun(@(i) FitnessFunction(population(i,:)), 1:numPopulation)', ...
    50, 'b', 'filled'); % Graficar los individuos
title('GA - Movimiento de Individuos');
xlabel('x'); ylabel('y'); zlabel('f(x, y)');
grid on;

%% --- Bucle de iteración ---
for iter = 1:maxIterations
    % --- PSO: Actualización ---
    for i = 1:numParticles
        % Evaluar la función objetivo en la posición actual
        currentScore = FitnessFunction(positions(i, :));
        % Actualizar el mejor personal si se encuentra un mejor puntaje
        if currentScore < pBestScores(i)
            pBestScores(i) = currentScore;
            pBest(i, :) = positions(i, :);
        end
        % Actualizar el mejor global si es necesario
        if currentScore < gBestScore
            gBestScore = currentScore;
            gBest = positions(i, :);
        end
    end
    
    % Actualizar posiciones y velocidades en PSO
    for i = 1:numParticles
        r1 = rand(1, numDimensions); % Coeficiente aleatorio para el mejor personal
        r2 = rand(1, numDimensions); % Coeficiente aleatorio para el mejor global
        velocities(i, :) = w * velocities(i, :) + ...
                           c1 * r1 .* (pBest(i, :) - positions(i, :)) + ...
                           c2 * r2 .* (gBest - positions(i, :)); % Actualizar velocidad
        positions(i, :) = positions(i, :) + velocities(i, :); % Actualizar posición
        positions(i, :) = max(min(positions(i, :), xMax), xMin); % Restringir al espacio de búsqueda
    end
    
    % Actualizar la gráfica de PSO
    subplot(1, 2, 1);
    scatter3(positions(:, 1), positions(:, 2), ...
        arrayfun(@(i) FitnessFunction(positions(i,:)), 1:numParticles)', ...
        50, 'r', 'filled'); % Graficar las partículas actualizadas
    drawnow; % Actualizar la gráfica en tiempo real
    
    % --- GA: Evolución ---
    scores = arrayfun(@(i) FitnessFunction(population(i,:)), 1:numPopulation)'; % Evaluar la población
    [~, sortedIdx] = sort(scores); % Ordenar la población por puntaje
    bestIndividuals = population(sortedIdx(1:round(numPopulation/2)), :); % Seleccionar los mejores individuos
    offspring = bestIndividuals + mutationRate * randn(size(bestIndividuals)); % Crear descendientes con mutación
    population = [bestIndividuals; offspring]; % Actualizar población
    population = max(min(population, xMax), xMin); % Restringir al espacio de búsqueda
    
    % Actualizar la gráfica de GA
    subplot(1, 2, 2);
    scatter3(population(:, 1), population(:, 2), ...
        arrayfun(@(i) FitnessFunction(population(i,:)), 1:numPopulation)', ...
        50, 'b', 'filled'); % Graficar los individuos actualizados
    drawnow; % Actualizar la gráfica en tiempo real
    
    % Almacenar el mejor resultado de GA
    gaBestScores(iter) = min(scores);
end
