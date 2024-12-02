% Autor: Luis Eduardo Arenas Deseano
% Proyecto: Sistemas Inteligentes
% Descripción: Ploteo de funcion objetivo
% Fecha: 1/12/2024

function plotobjective(fcn,range)
%PLOTOBJECTIVE plot a fitness function in two dimensions
%   plotObjective(fcn,range) where range is a 2 by 2 matrix in which each
%   row holds the min and max values to range over in that dimension.

%   Copyright 2003-2004 The MathWorks, Inc.

if(nargin == 0)
    fcn = @rastriginsfcn;
    range = [-5,5;-5,5];
end

pts = 100;
span = diff(range')/(pts - 1);
x = range(1,1): span(1) : range(1,2);
y = range(2,1): span(2) : range(2,2);

pop = zeros(pts * pts,2);
k = 1;
for i = 1:pts
    for j = 1:pts
        pop(k,:) = [x(i),y(j)];
        k = k + 1;
    end
end

values = feval(fcn,pop);
values = reshape(values,pts,pts);

surf(x,y,values)
shading interp
light
lighting phong
hold on
contour(x,y,values)
view(37,60)
hold off
end