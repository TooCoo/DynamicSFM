//include some things

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2\imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

//#include "opencv2\stereo.hpp"
#include "opencv2\calib3d.hpp"
#include "opencv2\calib3d\calib3d.hpp"

#include <eigen3/Eigen/Dense>

#include <iostream>
#include <fstream>
#include <string>

using namespace cv; // Might remove this later		**		**

int main() {

	//outline of what needs to be done:

	/*
		Load images - 2 of them for now

		Find Sift keypoints in each image

		Find matches between images

		Use matches to estimate potential motion models - maybe around 100
			This can be done using inbuilt openCV functions - RANSAC or 8 Points...

		Run model fitting algorithm
			this requires cost term between each point pair and model - the sampson error is a good candiadate
			Also needs neighboiurs of keypoints - use FLANN or similar

		I should now have segmented my keypoints into discrte motion models.
			So plot accordingly onto figure to show that the models have been correctly segmented

		?? Can I run a SFM based on each motion model and reconstruct 3D representations of each object...

		?? Can I store reconstructed objects to help fit future models...

	*/

	
	#pragma region Load Images
		
	//std::string imageName1("2_1m.jpg");
	//std::string imageName2("2_2m.jpg");
	
	//std::string imageName1("6_1m.png");
	//std::string imageName2("6_2m.png");

	//std::string imageName1("MY1s.jpg");
	//std::string imageName2("MY2s.jpg");

	//std::string imageName1("myC1s.jpg");
	//std::string imageName2("myC2s.jpg");
		
	std::string imageName2("blend_1.png");
	std::string imageName1("blend_2.png");


	//Load the images
	Mat i1, i2;
	i1 = imread(imageName1.c_str(), CV_LOAD_IMAGE_GRAYSCALE); // Read the file
	if (i1.empty())                      // Check for invalid input
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	i2 = imread(imageName2.c_str(), CV_LOAD_IMAGE_GRAYSCALE); // Read the file
	if (i2.empty())                      // Check for invalid input
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	//display the images
	namedWindow("Display window", WINDOW_NORMAL); // Create a window for display.
	resizeWindow("Display window", 800, 600);
	imshow("Display window", i1);                // Show our image inside it.

	//waitKey(0);

	#pragma endregion Loads two images

	#pragma region SIFT

	
	std::cout << "Begin SIFT...";

	//Find Sift keypoints in each image
	//cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();	

	//-- Step 1: Detect the keypoints:
	
	int 	nfeatures = 0;
	int 	nOctaveLayers = 3;
	double 	contrastThreshold = 0.1;
	double 	edgeThreshold = 10;
	double 	sigma = 1.6;
	Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create(0, 3, 0.07, 10, 1.6);
	
	//int minHessian = 400;
	//Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(minHessian);
	
	//vectors to hold the keypoints
	std::vector<KeyPoint> keypoints_1, keypoints_2;

	//find keypoints
	detector->detect(i1, keypoints_1);
	detector->detect(i2, keypoints_2);

	//-- Draw keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;

	drawKeypoints(i1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(i2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	
	namedWindow("Keypoints 1", WINDOW_NORMAL); // Create a window for display.
	resizeWindow("Keypoints 1", 800, 600);
	imshow("Keypoints 1", img_keypoints_1);                // Show our image inside it.

	namedWindow("Keypoints 2", WINDOW_NORMAL); // Create a window for display.
	resizeWindow("Keypoints 2", 800, 600);
	imshow("Keypoints 2", img_keypoints_2);                // Show our image inside it.
	
	//-- Step 2: Calculate descriptors (feature vectors)    
	Mat descriptors_1, descriptors_2;
	detector->compute(i1, keypoints_1, descriptors_1);
	detector->compute(i2, keypoints_2, descriptors_2);
	
	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	//-- PS.- radiusMatch can also be used here.
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_1.rows; i++)
	//for (int i = 0; i < 400; i++)
	{
		//pick the best matches within a threshold
		if (matches[i].distance <= max(4 * min_dist, 0.2))
		{
			good_matches.push_back(matches[i]);
		}
	}

	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches(i1, keypoints_1, i2, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Show detected matches
	namedWindow("Good Matches", WINDOW_NORMAL); // Create a window for display.
	resizeWindow("Good Matches", 1600, 600);
	imshow("Good Matches", img_keypoints_1);                // Show our image inside it.
	imshow("Good Matches", img_matches);

	int nGoodMatches = (int)good_matches.size();

	for (int i = 0; i < nGoodMatches; i++)
	{		
		printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].trainIdx, good_matches[i].queryIdx);
	}


	
	#pragma endregion Calculates features and correspondences, matches found using surf

	#pragma region Model_Estimation

	std::vector<Mat> Flist;
	int nF = 100;
	int countFails = 0;
	//for (int iF = 0; iF < nF; iF++) {
	for (int iF = 0; iF < nF;) {

		int nPointsToUse = 10;	//arbitrarily chosen // should probs use less

		DMatch theseMatches;
		int p1Ind;
		int p2Ind;

		std::vector<Point2f> p1(nPointsToUse);
		std::vector<Point2f> p2(nPointsToUse);

		//InputArray p1IA;

		//I could try and use the n closet points to formulate F

		///*

		for (int j = 0; j < nPointsToUse; j++) {

			theseMatches = good_matches[(int)(rand() % nGoodMatches)];

			p1Ind = theseMatches.queryIdx;
			p2Ind = theseMatches.trainIdx;

			p1[j] = keypoints_1[p1Ind].pt;
			p2[j] = keypoints_2[p2Ind].pt;


		}
		//*/

		/*
		std::vector<Point2f> pointsForSearch; //Insert all 2D points to this vector
		
		//put all point into search list
		for (int im = 0; im < nGoodMatches; im++) {
			pointsForSearch.push_back(keypoints_1[good_matches[im].trainIdx].pt);
		}

		flann::KDTreeIndexParams indexParams;
		flann::Index kdtree(Mat(pointsForSearch).reshape(1), indexParams);

		float range = 5000.0;

		theseMatches = good_matches[(int)(rand() % nGoodMatches)];
		p1Ind = theseMatches.trainIdx;
		
		Point2f queryP1 = keypoints_1[p1Ind].pt;		

		std::vector<float> query;
		query.push_back(float(queryP1.x)); //Insert the 2D point we need to find neighbours to the query
		query.push_back(float(queryP1.y)); //Insert the 2D point we need to find neighbours to the query
		std::vector<int> indices;
		std::vector<float> dists;
		kdtree.radiusSearch(query, indices, dists, range, nPointsToUse);
		
		//now I should have indicies which correspond to the neighbours

		for (int ind = 0; ind < indices.size(); ind++) {
			p1[ind] = keypoints_1[good_matches[indices[ind]].trainIdx].pt;
			p2[ind] = keypoints_2[good_matches[indices[ind]].queryIdx].pt;
		}
		*/

		//Mat F = findFundamentalMat(p1, p2, 3.0, 0.99, CV_FM_8POINT); // CV_FM_8POINT  CV_FM_7POINT  CV_FM_RANSAC
		//Mat F = Mat(3, 3, CV_64F);
		Mat F = findFundamentalMat(p1, p2, CV_FM_RANSAC, 3, 0.99);
		//Flist.push_back(F);
		
		bool success = true;

		try{
			F.at<double>(2, 1);			
		}
		catch (Exception e) {			
			success = false;
		}

		if (success) {
			iF++;			
			Flist.push_back(F);
		}
		else {
			countFails++;
		}
				
		std::cout << iF << "\n";
		
		//can I draw the epipolar lines for each F

		//or check F is correct using x2' F x1 = 0

	}

	//std::cout << "fail: " << countFails << "\n";
	//std::cout << "F size: " << Flist.size() << "\n";

	//system("PAUSE");
	
	#pragma endregion Now that I have my feature matches - I need to estimate my fundamental matrices
	
	#pragma region Model_fitting
		
	int nNeighbours = 10;

	Eigen::MatrixXd uniaryWeights = Eigen::MatrixXd::Zero(nF, nGoodMatches);

	Eigen::MatrixXd pointsGoodMatches1 = Eigen::MatrixXd::Zero(2, nGoodMatches);
	Eigen::MatrixXd pointsGoodMatches2 = Eigen::MatrixXd::Zero(2, nGoodMatches);

	Eigen::MatrixXd pointsMat1 = Eigen::MatrixXd::Zero(3, 1);
	Eigen::MatrixXd pointsMat2trans = Eigen::MatrixXd::Zero(1, 3);
	Eigen::MatrixXd Fmat = Eigen::MatrixXd::Zero(3, 3);				
		

	RNG rng(111);

	for (int iF = 0; iF < nF; iF++) {

		DMatch theseMatches;

		//std::cout << iF << "\n";
		//std::cout << Flist[iF] << "\n";
			

		//fill this 
		for (int di = 0; di < 3; di++) {

			Fmat(di, 0) = Flist[iF].at<double>(di, 0);
			Fmat(di, 1) = Flist[iF].at<double>(di, 1);
			Fmat(di, 2) = Flist[iF].at<double>(di, 2);

			//Fmat(0, di) = Flist[iF].at<double>(di, 0);
			//Fmat(1, di) = Flist[iF].at<double>(di, 1);
			//Fmat(2, di) = Flist[iF].at<double>(di, 2);
		}


		//change from points to use to all matches when calculating the uniary terms
		//for (pi = 0; pi < nPointsToUse; pi++) {
		for (int pi = 0; pi < nGoodMatches; pi++) {

			theseMatches = good_matches[pi];

			int p1Ind = theseMatches.trainIdx;
			int p2Ind = theseMatches.queryIdx;

			pointsMat1(0, 0) = keypoints_1[p1Ind].pt.x;
			pointsMat1(1, 0) = keypoints_1[p1Ind].pt.y;
			pointsMat1(2, 0) = 1.0;

			pointsMat2trans(0, 0) = keypoints_2[p2Ind].pt.x;
			pointsMat2trans(0, 1) = keypoints_2[p2Ind].pt.y;
			pointsMat2trans(0, 2) = 1.0;

			Eigen::MatrixXd e1 = pointsMat2trans * Fmat;
			Eigen::MatrixXd e2 = pointsMat1.transpose() * Fmat.transpose();

			Eigen::MatrixXd result = pointsMat2trans * Fmat * pointsMat1;

			float A = e1(0, 0);
			float B = e1(0, 1);
			float C = e1(0, 2);

			float x1 = keypoints_1[p1Ind].pt.x;
			float y1 = keypoints_1[p1Ind].pt.y;

			float d = A*x1 + B * y1 + C;

			if (d < 0) d *= -1.0f;

			d = d / sqrtf(A*A + B*B);

			//std::cout << "d: " << d << "\n";
			Point3d point1 = Point3f(keypoints_1[good_matches[pi].trainIdx].pt.x, keypoints_1[good_matches[pi].trainIdx].pt.y, 1.0);
			Point3d point2 = Point3f(keypoints_2[good_matches[pi].queryIdx].pt.x, keypoints_2[good_matches[pi].queryIdx].pt.y, 1.0);
			Mat pt1(point1);
			Mat pt2(point2);
			//std::cout << Flist[0].type() << "\n";
			//std::cout << sampsonDistance(pt1, pt2, Flist[iF]) << "\n";
			//uniaryWeights(iF, pi) = d;
			uniaryWeights(iF, pi) = sampsonDistance(pt1, pt2, Flist[iF]);
			//system("PAUSE");

		}
	}

	//neighbours

	//run optimisation

	
	#pragma endregion I now have nF potential fundamental matrices I need to optimise

	#pragma region findNeighbours

	// fill neighbours, and neighbourCost

	nNeighbours = 5;
	Eigen::MatrixXd neighbours = Eigen::MatrixXd::Zero(nNeighbours, nGoodMatches);
	Eigen::MatrixXd neighbourCost = Eigen::MatrixXd::Zero(nNeighbours, nGoodMatches);

	std::vector<Point2f> pointsForSearch; //Insert all 2D points to this vector

	for (int i = 0; i < nGoodMatches; i++) {
		pointsForSearch.push_back(keypoints_1[good_matches[i].trainIdx].pt);
	}

	flann::KDTreeIndexParams indexParams(20);
	flann::Index kdtree(Mat(pointsForSearch).reshape(1), indexParams);	

	float range = 5000.0;
		

	for (int i = 0; i < nGoodMatches; i++) {			
			
		pointsGoodMatches1(0, i) = keypoints_1[good_matches[i].trainIdx].pt.x;
		pointsGoodMatches1(1, i) = keypoints_1[good_matches[i].trainIdx].pt.y;

		pointsGoodMatches2(0, i) = keypoints_2[good_matches[i].queryIdx].pt.x;
		pointsGoodMatches2(1, i) = keypoints_2[good_matches[i].queryIdx].pt.y;

		std::vector<float> query;
		query.push_back((keypoints_1[good_matches[i].trainIdx].pt.x)); //Insert the 2D point we need to find neighbours to the query
		query.push_back((keypoints_1[good_matches[i].trainIdx].pt.y)); //Insert the 2D point we need to find neighbours to the query
		std::vector<int> indices;
		std::vector<float> dists;

		

		//kdtree.radiusSearch(query, indices, dists, range, nNeighbours);
		kdtree.knnSearch(query, indices, dists, nNeighbours, flann::SearchParams(64));


		//now I should have indicies which correspond to the neighbours
			
		for (int ind = 0; ind < indices.size(); ind++) {			
			neighbours(ind, i) = indices[ind];		//!!!		!!!			!!!
			neighbourCost(ind, i) = dists[ind];
			std::cout << "i: " << indices[ind] << " d: " << dists[ind] << "\n";

			

		}

	}
	#pragma endregion I have filled my weight matrix - now find the neighbours

	#pragma region SaveWeights

	std::string modelWeightsFileName = "FWeights";

	std::ofstream fileFW;
	fileFW.open(modelWeightsFileName);
	fileFW << uniaryWeights;		
	fileFW.close();

	std::string neighbourFileName = "neighbours";

	std::ofstream fileNeighbours;
	fileNeighbours.open(neighbourFileName);
	fileNeighbours << neighbours;
	fileNeighbours.close();

	std::string neighbourCostFileName = "neighbourCost";

	std::ofstream fileNeighbourCost;
	fileNeighbourCost.open(neighbourCostFileName);
	fileNeighbourCost << neighbourCost;
	fileNeighbourCost.close();

	//I should also save my point loacations so I can plot the final segmentation

	std::string pointsFileName = "points1";

	std::ofstream filePoints;
	filePoints.open(pointsFileName);
	filePoints << pointsGoodMatches1;
	filePoints.close();

	#pragma endregion Save the weights to a file

	std::cout << "done\n";

	cv::waitKey(0); // wait before closing the window


	return 0;
}


