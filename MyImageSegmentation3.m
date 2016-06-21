close all; clear all; clc;

nF = 100;


%Load data from my SIFT matcher and F estimator

%Unary = textscan('FWeights', 'f');
fileID = fopen('FWeights2');
[A,countF] = fscanf(fileID, '%f');
fclose(fileID);

fileID = fopen('neighbours2');
[B,countN] = fscanf(fileID, '%d');
fclose(fileID);

fileID = fopen('neighbourCost2');
[C,countNC] = fscanf(fileID, '%f'); 
fclose(fileID);

fileID = fopen('points2');
[D,countP] = fscanf(fileID, '%f');
fclose(fileID);

nPoints = countP/2;
nNeighbours = countN/nPoints;

unary = reshape(A', nPoints, nF);
neighbourhood = reshape(B', nPoints, nNeighbours); neighbourhood = neighbourhood';
neighbourCost = reshape(C', nPoints, nNeighbours); neighbourCost = neighbourCost';
currentLabel = zeros(1, nPoints);
costOfModelApp = ones(1, nF) * 10; costOfModelApp = costOfModelApp';
outlier = 10.0;
points = reshape(D', nPoints, 2);

%neighbourhoodZero = neighbourhood == 0;

%neighbourCostMax = max(neighbourCost(:));
%unaryMax = max(unary(:));
%unary = unary * (1/unaryMax);
%unary = unary .* unary;
%unary = unary * 20;

%neighbourhood = neighbourhood + ones(size(neighbourhood)) - neighbourhoodZero;
%neighbourCost = ones(size(neighbourCost))*0.1;



for i = 1:nPoints
   for j = 1:nNeighbours
              
       if neighbourhood(j,i) == i
           neighbourhood(j, i) = 0;
       end
       
   end    
end

neighbourCost = ones(size(neighbourCost))*1;


rng(10);
cols = rand(nF, 3);
%cols = hsv(nF);

figure;
%img = imread('2_1m.jpg');
%img = rgb2gray(imread('6_1m.png'));
img = rgb2gray(imread('blend_1.png'));
%img = imread('myD1m.jpg');
imshow(img);


nDrawn = 0;


for j = 1:1

    [models internal]= expand(unary, neighbourhood, neighbourCost, currentLabel, costOfModelApp, outlier);

    hold on;
    for i = 1:nPoints    
        %plot(points(i,1),points(i,2),'r.','MarkerSize',20)
        if(internal(i,1) ~= 0)
            nDrawn = nDrawn ;
            plot(points(i,1),points(i,2),'r.', 'Color', cols(internal(i, 1), :),'MarkerSize',20);
        end
    end
    hold off;

   

 end








