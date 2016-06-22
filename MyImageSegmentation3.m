close all; clear all; clc;

nF = 100;


%Load data from my SIFT matcher and F estimator

%Unary = textscan('FWeights', 'f');
fileID = fopen('FWeights');
[A,countF] = fscanf(fileID, '%f');
fclose(fileID);

fileID = fopen('neighbours');
[B,countN] = fscanf(fileID, '%d');
fclose(fileID);

fileID = fopen('neighbourCost');
[C,countNC] = fscanf(fileID, '%f'); 
fclose(fileID);

fileID = fopen('points1');
[D,countP] = fscanf(fileID, '%f');
fclose(fileID);

nPoints = countP/2;
nNeighbours = countN/nPoints;

unary = reshape(A', nPoints, nF);
neighbourhood = reshape(B', nPoints, nNeighbours); neighbourhood = neighbourhood';
neighbourCost = reshape(C', nPoints, nNeighbours); neighbourCost = neighbourCost';
currentLabel = zeros(1, nPoints);
costOfModelApp = ones(1, nF) * 0.5; costOfModelApp = costOfModelApp';
outlier = 15.0;
points = reshape(D', nPoints, 2);

%neighbourhoodZero = neighbourhood == 0;

neighbourCostMax = max(neighbourCost(:));
%unaryMax = max(unary(:));
%unary = unary * (1/unaryMax);
%unary = unary .* unary;
%unary = unary * 20;

%neighbourhood = neighbourhood + ones(size(neighbourhood)) - neighbourhoodZero;
%neighbourCost = ones(size(neighbourCost))*0.1;

neighbourCost = ones(size(neighbourCost))*2;

%max is around 5k


max_d = 5000;

for i = 1:nPoints
   for j = 1:nNeighbours
       
       %neighbourCost(j,i) = neighbourCost(j,i);
       
       if neighbourhood(j,i) == i
           neighbourhood(j, i) = 0;
       end
       
   end    
end


%figure;
%plot(neighbourCost(:,1), '.r');

rng(10);
cols = rand(nF, 3);
%cols = hsv(nF);

figure;
%img = imread('2_1m.jpg');
%img = rgb2gray(imread('6_1m.png'));
img = rgb2gray(imread('blend_2.png'));
%img = imread('myD1m.jpg');
imshow(img);


nDrawn = 0;


for j = 1:1

    [models internal]= expand(unary, neighbourhood, neighbourCost, currentLabel, costOfModelApp, outlier);

    hold on;
    for i = 1:nPoints    
        %plot(points(i,1),points(i,2),'r.','MarkerSize',20)
        if(internal(i,1) ~= 0)
            %nDrawn = nDrawn ;
            plot(points(i,1),points(i,2),'r.', 'Color', cols(internal(i, 1), :),'MarkerSize',20);
        end
    end
    hold off;

   

end

 %lets check that the neighbours are indeed the neighbours
 figure;
 imshow(img);
 for i = 1:nPoints
     imshow(img);
     hold on;
     
     for j = 1:nNeighbours
     
         index = neighbourhood(j,i);
         
         if index ~=0

             x = points(neighbourhood(j,i), 1);
             y = points(neighbourhood(j,i), 2);


             plot(x, y,'r.', 'Color', cols(internal(i, 1), :),'MarkerSize',20);

         end
     
     end
     
     hold off;
     pause;
end








