#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;

void displayImage(string window_name, cv::Mat image) {
    namedWindow(window_name, WINDOW_AUTOSIZE );
    imshow(window_name, image);
}

int main(int argc, char** argv ) {

    cv::Mat img_r, img_l;
    img_r = cv::imread("../data/carR.png",cv::IMREAD_GRAYSCALE);
    img_l = cv::imread("../data/carL.png",cv::IMREAD_GRAYSCALE);
    
    // Define keypoints vector
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    
    // Define feature detector
    cv::Ptr<cv::Feature2D> ptrFeature2D = cv::xfeatures2d::SIFT::create(74);
    
    // Keypoint detection
    ptrFeature2D->detect(img_r,keypoints1);
    ptrFeature2D->detect(img_l,keypoints2);
    
    // Extract the descriptor
    cv::Mat descriptors1;
    cv::Mat descriptors2;
    
    ptrFeature2D->compute(img_r,keypoints1,descriptors1);
    ptrFeature2D->compute(img_l,keypoints2,descriptors2);
    
    // Construction of the matcher
    cv::BFMatcher matcher(cv::NORM_L2);
    
    // Match the two image descriptors
    std::vector<cv::DMatch> outputMatches;
    matcher.match(descriptors1,descriptors2, outputMatches);

    cv::Mat matchImage;
    cv::namedWindow("Matched Image");
    cv::drawMatches(img_r, keypoints1, img_l, keypoints2, outputMatches, matchImage, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite("../output/matches.jpg", matchImage);

    // Convert keypoints into Point2f
    std::vector<cv::Point2f> points1, points2;
    for (std::vector<cv::DMatch>::const_iterator it=   
        outputMatches.begin(); it!= outputMatches.end(); ++it) {    
        
        // Get the position of left keypoints
        points1.push_back(keypoints1[it->queryIdx].pt);
        // Get the position of right keypoints
        points2.push_back(keypoints2[it->trainIdx].pt);
    }

    std::vector<uchar> inliers(points1.size(),0);
    cv::Mat fundamental= cv::findFundamentalMat(
                        points1,points2, // matching points
                        inliers,         // match status (inlier or outlier)  
                        cv::FM_RANSAC,   // RANSAC method
                        1.0,        // distance to epipolar line
                        0.98);     // confidence probability
    
    cout<< fundamental << endl; //include this for seeing fundamental matrix

    // Compute homographic rectification
    cv::Mat h1, h2;
    cv::stereoRectifyUncalibrated(points1, points2, fundamental,
                                  img_r.size(), h1, h2);

    // Rectify the images through warping
    cv::Mat rectified1;
    cv::warpPerspective(img_r, rectified1, h1, img_r.size());
    cv::Mat rectified2;
    cv::warpPerspective(img_l, rectified2, h2, img_l.size());

    // Compute disparity
    cv::Mat disparity;
    cv::Ptr<cv::StereoMatcher> pStereo = cv::StereoSGBM::create(0,32,5);
    pStereo->compute(rectified1, rectified2, disparity);
    
    cout << "Disparity Image saved under output folder" << endl;
    cv::imwrite("../output/disparity.jpg", disparity);
    displayImage("Disparity Image", disparity);
    cout << "Press esc to exit" << endl;
    waitKey(0);

    return 0;
}