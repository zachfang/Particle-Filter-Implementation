/**
 * @file   	particle_filter.h
 * @author 	Hsu-Chieh Hu, Ming Hsiao, Ching-Hsin (Zach) Fang
 * @date   	04/27/2016
 * @brief  	An implementation of particle filter using laser scan and odometry mounted on the robot. 
 			With a global map generated beforehand, the robot localizes itself based on its odometry 
 			and laser scan data.  
 */

#include <string>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <random>
#include <limits> 
#include <math.h> 
#include <iomanip>  
#include <eigen3/Eigen/Dense>
#include "opencv2/opencv.hpp"

/**
 * @brief	The size of the map 
 */
const int kMapX = 800;
const int kMapY = 800;

/**
 * @brief	The number of particles sampling every iteration
 */
const int kSampleNum = 30000;

/**
 * @brief	The uncertainty of x, y and theta 
 */
const double kVarianceUV = 0.05;
const double kVarianceR = 0.01;

/**
 * @brief	Constant epsilon used to compare two double values 
 */
const double kNumerical = std::numeric_limits<double>::epsilon();

/**
 * @brief	A wrapper to represent the global map and free space for particles. Define free space, it's a 
 			std::vector stores points in the global map that is not occupied. 	
 */
struct MapWrapper{
	Eigen::MatrixXd global_map_;
	std::vector<Eigen::Vector2d> free_space_; 
};

/**
 * @brief	A wrapper to represent particles including properties like position, weighting, uncertainty.  	
 */
struct ParticlesWrapper{
	Eigen::MatrixXd particles_;
	Eigen::MatrixXd sigma_;
	Eigen::VectorXd weights_;
	Eigen::VectorXd prev_weights_; 
};


/**
 * @brief	Parses the map from a data file  	
 * @param1	The map data file to read from
 * @param2  A Eigen::MatrixXd to store the ocuppancy status 
 */
void parseMap(std::ifstream *, Eigen::MatrixXd&);

/**
 * @brief	Parses each line of the map data and assigns value into matrix. It is called by parseMap().  	
 * @param1	Content of each line
 * @param2	Row index of the matrix
 * @param3	A Eigen::MatrixXd to store the ocuppancy status   
 */
void parseMapLine(const std::string&, const int&, Eigen::MatrixXd&);

/**
 * @brief	Initializes the particles at first iteration. 
 * @param1	Global map matrix indicating ocupancy status
 * @param2	Free space of the map
 * @param3	A matrix to store particle location(x, y, theta)   
 */
void initParticles(const Eigen::MatrixXd&, std::vector<Eigen::Vector2d>&, Eigen::MatrixXd&);

/**
 * @brief	Generates a cv::Mat from global_map_ eigen matrix	
 * @param1	Global map matrix indicating ocupancy status
 * @param2	cv::Mat to store result
 */
void initImage(const Eigen::MatrixXd&, cv::Mat&);

/**
 * @brief	Ranodmly generates a set of postion, and theta value	
 * @param1	A container for stroing the result
 * @param2	Free space std::vector of the global map
 */
void randomAssignXYTheta(Eigen::Vector3d&, const std::vector<Eigen::Vector2d>&);

/**
 * @brief	Use opencv function to draw particle on the image for visualization	
 * @param1	cv::Mat object to draw
 * @param2	The particle wrapper. Getting particle location from it
 */
void drawParticles(cv::Mat&, ParticlesWrapper*);

/**
 * @brief	Parses log file to get odometry data and laser scan data at each iteration	
 * @param1	Content of each line of log file
 * @param2	A std::vector to store the result   
 */
void parseLog(const std::string&, std::vector<std::string>&);

/**
 * @brief	With the processed result from previous function, assigns value to odometry, motion between frames and laser scan	
 * @param1	A std::vector storing processed data
 * @param2	Container to store odometry 
 * @param3	Container to store motion between t and t+1 frame
   @param4	Container to store laser scan data   
 */
void getDataFromLog(const std::vector<std::string>&, Eigen::Vector3d&, Eigen::Vector3d&, Eigen::VectorXd&);

/**
 * @brief	Applies motion on each randomly generated particles and evaluates the similiraty function	
 * @param1	The particle wrapper 
 * @param2	The motion calculated from previous function   
 */
void applyMotion(ParticlesWrapper *, const Eigen::Vector3d&);

/**
 * @brief	Calculates the loss function from laser scan data	
 * @param1	The index of current particle
 * @param2	A double to store final result
 * @param3	Laser scan data
 * @param4	The particle wrapper
 * @param5	The map wrapper   
 */
void calculateLoss(const int&, double&, const Eigen::VectorXd&, MapWrapper*, ParticlesWrapper*);

/**
 * @brief	Updates the weight_ and assigns prev_weights_ for all particles	
 * @param1	The particle wrapper
 * @param2	The map wrapper
 * @param3	Laser scan data   
 */
void updateParticlesWeights(ParticlesWrapper *, MapWrapper*, const Eigen::VectorXd&);

/**
 * @brief	Resamples particles based on current weighting and uncertainty of each particle
 * @param1	The map wrapper
 * @param2	Laser scan data   
 */
void resampleParticles(ParticlesWrapper*, MapWrapper*);