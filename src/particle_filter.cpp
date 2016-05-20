/**
 * @file   	particle_filter.cpp
 * @author 	Hsu-Chieh Hu, Ming Hsiao, Ching-Hsin (Zach) Fang
 * @date   	04/27/2016
 * @brief  	An implementation of particle filter using laser scan and odometry mounted on the robot.
 */

#include "particle_filter.h"

int main(int argc, char** argv){

	int time_stamp = 0;	
	cv::Mat map_image = cv::Mat::zeros(kMapX, kMapY, CV_8UC1);
	cv::Mat output_image = cv::Mat::zeros(kMapX, kMapY, CV_8UC1);
	
    std::cout<< " test1" << std::endl;
	if(argc < 3){
		std::cout<< "Please specify map data and log file..." << std::endl;
		return -1;
	}

	std::ifstream map_data(argv[1]);
	std::ifstream log_file(argv[2]);	

	// initialize matrix/vector for structs
	// using a matrix to represent the global map
	Eigen::MatrixXd global_map = Eigen::MatrixXd::Zero(kMapX, kMapY);
	std::vector<Eigen::Vector2d> free_space;
	Eigen::MatrixXd particles = Eigen::MatrixXd::Zero(kSampleNum, 3);
	Eigen::MatrixXd sigma = Eigen::MatrixXd::Zero(kSampleNum, 2);
	Eigen::VectorXd weights = Eigen::VectorXd::Constant(kSampleNum, 1.0);
	Eigen::VectorXd prev_weights = Eigen::VectorXd::Constant(kSampleNum, 1.0); 
	
	struct ParticlesWrapper *particle_wrapper = new ParticlesWrapper;
	*particle_wrapper = {particles, sigma, weights, prev_weights};
	struct MapWrapper *map_wrapper = new MapWrapper;
	*map_wrapper = {global_map, free_space};

	// data structures to hold robot motion and observation reading
	Eigen::Vector3d motion;
	Eigen::Vector3d prev_odom(0.0, 0.0, 999.0);
	Eigen::VectorXd laser_data = Eigen::VectorXd::Zero(180, 1);	

	// initialize particle filter
	parseMap(&map_data, map_wrapper->global_map_);
	initParticles(map_wrapper->global_map_, map_wrapper->free_space_, particle_wrapper->particles_);
	initImage(map_wrapper->global_map_, map_image);

	output_image = map_image.clone();
	drawParticles(output_image, particle_wrapper);	

	// store each reading from log file
	std::string tmp_log;

	// start partice filter iteration
	while(getline(log_file, tmp_log)){

		// a vector to hole processed substring
		std::vector<std::string> log_data;
		parseLog(tmp_log, log_data);

		if(log_data[0]== "L"){

			getDataFromLog(log_data, prev_odom, motion, laser_data);
			applyMotion(particle_wrapper, motion);
			updateParticlesWeights(particle_wrapper, map_wrapper, laser_data);
			resampleParticles(particle_wrapper, map_wrapper);
			
			output_image = map_image.clone();
			drawParticles(output_image, particle_wrapper);

			// generate image for each frame (cv::VideoWriter not working for me for unknown reasons)
			std::ostringstream ss;
			ss << std::setw(4) << std::setfill('0') << time_stamp;
			std::string img_name = "./../output/" + ss.str() + ".jpg";			
			
			imwrite(img_name, output_image);
			
			if(cv::waitKey(50) >= 0) break;
			time_stamp++;
		}			
	}
	
	delete particle_wrapper;
	delete map_wrapper;
	return 0;
}

void parseMap(std::ifstream *data, Eigen::MatrixXd& map){

	std::string tmp_str;
	int data_line_number = 0;
	int matrix_line_number = 0;
	// the first seven lines are no use
	int skip_line_number = 7;    
	
	while(getline(*data, tmp_str)){
		if(data_line_number >= skip_line_number){
			parseMapLine(tmp_str, matrix_line_number++, map);
		}
		data_line_number++;
	}
}

void parseMapLine(const std::string& tmp_str, const int& line_num, Eigen::MatrixXd& map){

	char delimiter = ' ';
	std::string::size_type start = 0;
	std::string::size_type end = tmp_str.find(delimiter);
	std::vector<double> vec(kMapX, 0.0);	
	int counter = 0;
	
	// use start and end to split the string and store into the vector
	while(end != std::string::npos){
		// for unknown reason, atof() is more stable than std::stod()
		vec[counter++] = atof(tmp_str.substr(start, end-start).c_str());
		start = end+1;
		end = tmp_str.find(delimiter, start);		
	}
	
	// assign values back to the vector
	for(size_t i=0; i<kMapX; i++){
		map(line_num, i) = vec[i]; 
	}
}

void initParticles(const Eigen::MatrixXd& global_map, std::vector<Eigen::Vector2d>& free_space, Eigen::MatrixXd& particles){

	// traverse the map and put all the free space into a vector
	for(size_t i=0; i< kMapX; i++){
		for(size_t j=0; j< kMapY; j++){
			if(global_map(i, j) == 1){
				Eigen::Vector2d tmp((double)i, (double)j);
				free_space.push_back(tmp);
			}
		}
	}

	// generate random x, y, theta value for each particle
	for(size_t i=0; i< kSampleNum; i++){
		Eigen::Vector3d randomXYTheta;
		randomAssignXYTheta(randomXYTheta, free_space);
		particles.row(i) = randomXYTheta; 
	}
}

void randomAssignXYTheta(Eigen::Vector3d& vec, const std::vector<Eigen::Vector2d>& free_space){
	int seed = std::rand() % free_space.size();
	vec(0) = free_space[seed](0);
	vec(1) = free_space[seed](1);
	double seed_heading = static_cast<double> (std::rand()) / static_cast<double> (RAND_MAX) *2* M_PI;
	vec(2) = seed_heading;
}

void initImage(const Eigen::MatrixXd& global_map, cv::Mat& map_image){
	for(size_t i=0; i< kMapX; i++){
		for(size_t j=0; j< kMapY; j++){
			map_image.data[i* kMapX+ j] = global_map(i, j) * 255;
		}
	}
}

void drawParticles(cv::Mat& img, ParticlesWrapper* par){
	for(size_t i=0; i< kSampleNum; i++){
		if(par->prev_weights_(i) != 0){
			img.data[int(par->particles_(i, 0))* kMapX + int(par->particles_(i, 1))] = 127;
		}
	}

	cv::imshow("Particle flter output map", img);
}

void parseLog(const std::string& tmp_log, std::vector<std::string>& log_data){

	char delimiter = ' ';
	std::string::size_type start = 0;
	std::string::size_type end = tmp_log.find(delimiter);

	while(end != std::string::npos){
		log_data.push_back(tmp_log.substr(start, end-start));
		start = end+1;
		end = tmp_log.find(delimiter, start);

		if(end == std::string::npos){			
			log_data.push_back(tmp_log.substr(start, tmp_log.length()-start));
		}
	}
}

void getDataFromLog(const std::vector<std::string>& log_data, Eigen::Vector3d& prev_odom, Eigen::Vector3d& motion, Eigen::VectorXd& laser_data){
	
	// assign odometry
	Eigen::Vector3d cur_odom(std::stod(log_data[1]), std::stod(log_data[2]), std::stod(log_data[3]));

	// assign motion
	if(prev_odom(2) == 999.0){

		motion(0) = 0.0;
		motion(1) = 0.0;
		motion(2) = 0.0;
	} else {
		double dx = cur_odom(0) - prev_odom(0);
		double dy = cur_odom(1) - prev_odom(1);
		motion(2) = cur_odom(2) - prev_odom(2);
		motion(0) = ((dx* cos(-cur_odom(2))) - dy* sin(-cur_odom(2)))/10.0; 
		motion(1) = ((dx* sin(-cur_odom(2))) + dy* cos(-cur_odom(2)))/10.0; 
	}
	
	prev_odom(0)= cur_odom(0);
	prev_odom(1)= cur_odom(1);
	prev_odom(2)= cur_odom(2);

	// assign laser data
	for(size_t i=0; i< 180; i++){
		laser_data(i) = std::stod(log_data[i+7]);
	}
}

void applyMotion(ParticlesWrapper *par, const Eigen::Vector3d& motion){

	for(size_t i=0; i< kSampleNum; i++){
		
		double dx = motion(0) * cos(par->particles_(i, 2))- motion(1) * sin(par->particles_(i,2));
		double dy = motion(0) * sin(par->particles_(i, 2))+ motion(1) * cos(par->particles_(i,2));
		par->particles_(i, 0) = par->particles_(i, 0) + dx;
		par->particles_(i, 1) = par->particles_(i, 1) + dy; 
		par->particles_(i, 2) = par->particles_(i, 2) + motion(2);

		par->sigma_(i, 0) = par->sigma_(i, 0) + kVarianceUV;
		par->sigma_(i, 1) = par->sigma_(i, 1) + kVarianceR; 
	}
}

void updateParticlesWeights(ParticlesWrapper *par, MapWrapper* map, const Eigen::VectorXd& laser_data){

	for(size_t i=0; i< kSampleNum; i++){
		
		if(par->particles_(i, 0) <0 || par->particles_(i, 0) >= kMapX){
			par->weights_(i) = 0.0;
			continue;
		}

		if(par->particles_(i, 1) <0 || par->particles_(i, 1) >= kMapY){
			par->weights_(i) = 0.0;
			continue;	
		}

		double map_value = map->global_map_((int)par->particles_(i, 0), (int)par->particles_(i, 1));
		if(fabs(map_value - 1.0) > kNumerical) {
			par->weights_(i) = 0.0;
			continue;
		}	
	}

	for(size_t i= 0; i< kSampleNum; i++){

		if(fabs(par->weights_(i)- 0.0) < kNumerical){
			continue;
		} 
		
		double compare_result = 0;
		calculateLoss(i, compare_result, laser_data, map, par);
		par->weights_(i) = exp(-compare_result/1000.0);		
	}

	double sum_of_weight = par->weights_.sum();
	par->weights_ = par->weights_ / (sum_of_weight);
	//par->prev_weights_ = par->weights_;
}

void calculateLoss(const int& par_idx, double& result, const Eigen::VectorXd& laser_data, MapWrapper* map, ParticlesWrapper* par){

	int resolution = 10;
	size_t num_comparision = 180/ resolution;
	double angle_rad = -M_PI/2.0 + M_PI/360.0;	
	Eigen::Vector3d single_particle = par->particles_.row(par_idx);
	
	for(size_t i=0; i< num_comparision; i++){

		double single_laser_dir = laser_data(i* resolution);
		double dx = 1.0* cos(angle_rad + single_particle(2));
    	double dy = 1.0* sin(angle_rad + single_particle(2));
    	double ray_x = single_particle(0);
    	double ray_y = single_particle(1);
    	double ray_len = 0.0;

    	while (ray_x >= 0 && ray_x < kMapX && ray_y >=0 && ray_y < kMapY) {
        	if (map->global_map_((int)ray_x, (int)ray_y) < 0.1) {
            	break;
        	} else {
            	ray_x += dx;
            	ray_y += dy;
            	ray_len += 1.0;
        	}
    	}
    	result += fabs(ray_len - single_laser_dir/10.0);
    	angle_rad += (M_PI/180 * resolution);
	}
}

void resampleParticles(ParticlesWrapper* par, MapWrapper* map){
       
    std::default_random_engine generator;
    Eigen::VectorXd normailized_weights = par->weights_ * kSampleNum;

    Eigen::MatrixXd prev_particles = par->particles_;

    std::normal_distribution<double> normal(0.0,1.0);

    double cumulative_weights = 0;
    int count = 0;

    for (size_t i = 0; i < kSampleNum; i++){
    	// determine how many sample should come from previous weighting
        cumulative_weights += normailized_weights(i);
        while (count < (int)(cumulative_weights) && count <= kSampleNum){
            // generate new sample with normal distributed fluctuation
            double dx = normal(generator);
            double dy = normal(generator);
            double dTheta = normal(generator);
            Eigen::Vector3d temp;
            temp(0) = prev_particles(i,0) + par->sigma_(i,0) * dx; 
            temp(1) = prev_particles(i,1) + par->sigma_(i,0) * dy; 
            temp(2) = prev_particles(i,2) + par->sigma_(i,1) * dTheta;

            // check if the new particle is valid or redo it again 
            if((temp(0) >= 0) && (temp(0)< kMapX) && (temp(1) >= 0) && (temp(1) < kMapY)) {
                if(map->global_map_((int)temp(0),(int)temp(1)) == 1.0) {
                    par->particles_(count, 0) = temp(0);
                    par->particles_(count, 1) = temp(1);
                    par->particles_(count, 2) = temp(2);
                    count++;
                }
            }
        }
    }

    par->prev_weights_ = par->weights_;
    par->weights_ = Eigen::VectorXd::Constant(kSampleNum, 1.0);
    par->sigma_ = Eigen::MatrixXd::Zero(kSampleNum,2);
}
