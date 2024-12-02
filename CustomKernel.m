% Autor: Luis Eduardo Arenas Deseano
% Proyecto: Sistemas Inteligentes
% Descripción: Prueba de Custom Kernel
% Fecha: 1/12/2024


rng(1);  % For reproducibility
n = 100; % Number of points per quadrant

r1 = sqrt(rand(2*n,1));                     % Random radii
t1 = [pi/2*rand(n,1); (pi/2*rand(n,1)+pi)]; % Random angles for Q1 and Q3
X1 = [r1.*cos(t1) r1.*sin(t1)];             % Polar-to-Cartesian conversion

r2 = sqrt(rand(2*n,1));
t2 = [pi/2*rand(n,1)+pi/2; (pi/2*rand(n,1)-pi/2)]; % Random angles for Q2 and Q4
X2 = [r2.*cos(t2) r2.*sin(t2)];

X = [X1; X2];        % Predictors 
Y = ones(4*n,1);
Y(2*n + 1:end) = -1; % Labels
%%
% Plot the data.
figure;
gscatter(X(:,1),X(:,2),Y);
title('Scatter Diagram of Simulated Data')
%%
% Write a function that accepts two matrices in
% the feature space as inputs, and transforms them into a Gram matrix using
% the sigmoid kernel.
%
% <include>mysigmoid.m</include>
%
%%
% Save this code as a file named |mysigmoid| on your MATLAB(R)
% path.
%%
% Train an SVM classifier using the sigmoid kernel function. It is good
% practice to standardize the data.
Mdl1 = fitcsvm(X,Y,'KernelFunction','mysigmoid','Standardize',true);
%%
% |Mdl1| is a |ClassificationSVM| classifier containing the estimated
% parameters.
%%
% Plot the data, and identify the support vectors and the decision boundary.

    % Compute the scores over a grid
d = 0.02; % Step size of the grid
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];        % The grid
[~,scores1] = predict(Mdl1,xGrid); % The scores

figure;
h(1:2) = gscatter(X(:,1),X(:,2),Y);
hold on
h(3) = plot(X(Mdl1.IsSupportVector,1),...
    X(Mdl1.IsSupportVector,2),'ko','MarkerSize',10);
    % Support vectors
contour(x1Grid,x2Grid,reshape(scores1(:,2),size(x1Grid)),[0 0],'k');
    % Decision boundary
title('Scatter Diagram with the Decision Boundary')
legend({'-1','1','Support Vectors'},'Location','Best');
hold off
%%
% You can adjust the kernel parameters in an attempt to improve the shape
% of the decision boundary. This might also decrease the within-sample
% misclassification rate, but, you should first determine the out-of-sample
% misclassification rate.
%%
% Determine the out-of-sample misclassification rate by using 10-fold cross
% validation.
CVMdl1 = crossval(Mdl1);
misclass1 = kfoldLoss(CVMdl1);
misclass1
%%
% The out-of-sample misclassification rate is 13.5%.
%%
% Write another sigmoid function, but Set |gamma = 0.5;|.
%
% <include>mysigmoid2.m</include>
%%% Train SVM Classifier Using Custom Kernel
%%
% Save this code as a file named |mysigmoid2| on your MATLAB(R)
% path.
%%
% Train another SVM classifier using the adjusted sigmoid kernel. Plot the
% data and the decision region, and determine the out-of-sample
% misclassification rate.
Mdl2 = fitcsvm(X,Y,'KernelFunction','mysigmoid2','Standardize',true);
[~,scores2] = predict(Mdl2,xGrid);

figure;
h(1:2) = gscatter(X(:,1),X(:,2),Y);
hold on
h(3) = plot(X(Mdl2.IsSupportVector,1),...
    X(Mdl2.IsSupportVector,2),'ko','MarkerSize',10);
title('Scatter Diagram with the Decision Boundary')
contour(x1Grid,x2Grid,reshape(scores2(:,2),size(x1Grid)),[0 0],'k');
legend({'-1','1','Support Vectors'},'Location','Best');
hold off

CVMdl2 = crossval(Mdl2);
misclass2 = kfoldLoss(CVMdl2);
misclass2
