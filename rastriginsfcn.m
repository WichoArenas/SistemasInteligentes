% Autor: Luis Eduardo Arenas Deseano
% Proyecto: Sistemas Inteligentes
% Descripción: Funcion de Rastringin
% Fecha: 1/12/2024

function scores = rastriginsfcn(pop)
%RASTRIGINSFCN Compute the "Rastrigin" function.

%   Copyright 2003-2004 The MathWorks, Inc.


    % pop = max(-5.12,min(5.12,pop));
    scores = 10.0 * size(pop,2) + sum(pop .^2 - 10.0 * cos(2 * pi .* pop),2);