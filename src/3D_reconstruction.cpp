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

cv::Vec3d triangulatePoint(const cv::Mat &p1, const cv::Mat &p2, const cv::Vec2d &u1, const cv::Vec2d &u2) {

  // system of equations assuming image=[u,v] and X=[x,y,z,1]
  // from u(p3.X)= p1.X and v(p3.X)=p2.X
  cv::Matx43d A(u1(0)*p1.at<double>(2, 0) - p1.at<double>(0, 0),
  u1(0)*p1.at<double>(2, 1) - p1.at<double>(0, 1),
  u1(0)*p1.at<double>(2, 2) - p1.at<double>(0, 2),
  u1(1)*p1.at<double>(2, 0) - p1.at<double>(1, 0),
  u1(1)*p1.at<double>(2, 1) - p1.at<double>(1, 1),
  u1(1)*p1.at<double>(2, 2) - p1.at<double>(1, 2),
  u2(0)*p2.at<double>(2, 0) - p2.at<double>(0, 0),
  u2(0)*p2.at<double>(2, 1) - p2.at<double>(0, 1),
  u2(0)*p2.at<double>(2, 2) - p2.at<double>(0, 2),
  u2(1)*p2.at<double>(2, 0) - p2.at<double>(1, 0),
  u2(1)*p2.at<double>(2, 1) - p2.at<double>(1, 1),
  u2(1)*p2.at<double>(2, 2) - p2.at<double>(1, 2));

  cv::Matx41d B(p1.at<double>(0, 3) - u1(0)*p1.at<double>(2,3),
                p1.at<double>(1, 3) - u1(1)*p1.at<double>(2,3),
                p2.at<double>(0, 3) - u2(0)*p2.at<double>(2,3),
                p2.at<double>(1, 3) - u2(1)*p2.at<double>(2,3));

  // X contains the 3D coordinate of the reconstructed point
  cv::Vec3d X;
  // solve AX=B
  cv::solve(A, B, X, cv::DECOMP_SVD);
  return X;
}

// triangulate a vector of image points
void iterativeLineartriangulation(const cv::Mat &p1, const cv::Mat &p2, const std::vector<cv::Vec2d> &pts1,
                 const std::vector<cv::Vec2d> &pts2, std::vector<cv::Vec3d> &pts3D) {

  for (int i = 0; i < pts1.size(); i++) {

    pts3D.push_back(triangulatePoint(p1, p2, pts1[i], pts2[i]));
  }
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

	// For stereo images data with know intrensic matrix 
    cv::Matx33d cameraMatrix(9.842439e+02, 0.000000e+00, 6.900000e+02,
                             0.000000e+00, 9.808141e+02, 2.331966e+02,
                             0.000000e+00, 0.000000e+00, 1.000000e+00);
	
	// For stereo images with know distortion coefficients 
    double distCoeffsVals[5] = {-3.728755e-01,2.037299e-01,2.219027e-03, 1.383707e-03, -7.233722e-02};
    cv::Mat distCoeffs = cv::Mat(1,5,CV_64F, distCoeffsVals);

    // Find the essential between image 1 and image 2  
    double focal = 9.842439e+02;
    cv::Mat inliers;
    cv::Mat essential = cv::findEssentialMat(points1, points2, cameraMatrix,
                        cv::RANSAC, 0.9, 1.0, inliers);

    cout<<"Essential: " << essential<<endl;

    // recover relative camera pose from essential matrix
    cv::Mat rotation, translation;
    cv::recoverPose(essential, points1, points2, cameraMatrix, rotation, translation, inliers);
    cout<<"Rotation: " << rotation << endl;
    cout<<"Translation: " << translation << endl;

    // compose projection matrix from R,T
    cv::Mat projection2(3, 4, CV_64F); // the 3x4 projection matrix
    rotation.copyTo(projection2(cv::Rect(0, 0, 3, 3)));
    translation.copyTo(projection2.colRange(3, 4));

    // compose generic projection matrix
    cv::Mat projection1(3, 4, CV_64F, 0.); // the 3x4 projection matrix
    cv::Mat diag(cv::Mat::eye(3, 3, CV_64F));
    diag.copyTo(projection1(cv::Rect(0, 0, 3, 3)));
    
    // to contain the inliers
    std::vector<cv::Vec2d> inlierPts1;
    std::vector<cv::Vec2d> inlierPts2;
    
    // create inliers input point vector for triangulation
    for (int i = 0; i < inliers.rows; i++) {
        if (inliers.at<uchar>(i)) {
            inlierPts1.push_back(cv::Vec2d(points1[i].x, points1[i].y));
            inlierPts2.push_back(cv::Vec2d(points2[i].x, points2[i].y));
        }
    }

    // undistort and normalize the image points
    std::vector<cv::Vec2d> points1u;
    cv::undistortPoints(inlierPts1, points1u, cameraMatrix, distCoeffs);
    std::vector<cv::Vec2d> points2u;
    cv::undistortPoints(inlierPts2, points2u, cameraMatrix, distCoeffs);

    // Triangulation
    std::vector<cv::Vec3d> points3D;
    iterativeLineartriangulation(projection1, projection2, points1u, points2u, points3D);
    
    cout<< "3D points :"<< points3D.size()<<endl;

    viz::Viz3d window; //creating a Viz window
    //Displaying the Coordinate Origin (0,0,0)
    window.showWidget("coordinate", viz::WCoordinateSystem());
    window.setBackgroundColor(cv::viz::Color::black());
    //Displaying the 3D points in green
    window.showWidget("points", viz::WCloud(points3D, viz::Color::green()));
    window.spin();

    // viewing the image
    // imshow("Right Image", img_r);
    // imshow("Left Image", img_l);
    //waitKey(0);

    return 0;
}