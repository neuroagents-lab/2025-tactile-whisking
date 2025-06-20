/*
WHISKiT Physics Simulator
Copyright (C) 2019 Nadina Zweifel (SeNSE Lab)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef PARAMETERS_H
#define PARAMETERS_H

#define PI 3.1415927

#define LINK_THETA_MAX PI/4
#define LINK_PHI_MAX PI/4
#define LINK_ZETA_MAX PI/360

#include <string>
#include <vector>
#include "LinearMath/btVector3.h"
#include <boost/filesystem.hpp>

class Parameters{
public:
	Parameters();
	~Parameters();
	
	// integration parameters
	float HZ;
	float TIME_STOP;
	int SUBSTEPS;

	// simulation parameters
	int NUM_RATS;
	int DEBUG;
	int PRINT;
	int SAVE;
	int SAVE_VIDEO;
	
	// collision object type
	int OBJ_CONVEX_HULL;
	std::string OBJ_PATH;
	float OBJ_SCALE;
	float OBJ_SPEED;
	btVector3 OBJ_POS;
	btVector3 OBJ_ORI;
	float OBJ_THETA; // rotation about z axis
	float OBJ_MASS;

	// whisker model parameters
	std::string WHISKER_MODEL_DIR;
	std::string dir_param;
    std::vector<std::string> WHISKER_NAMES;
	std::vector<int> WHISKER_INDEX;
    float BLOW; // scale whisker diameter for visibility - ATTENTION: will affect dynamics!!!
	int NO_CURVATURE; // remove curvature for debugging
	int NO_MASS; 	  // set mass to zero for debugging
	int NO_WHISKERS;  // remove whiskers for debugging
	float NUM_LINKS_PER_MM;
	float RHO_BASE;
	float RHO_TIP;
	float E;
	float ZETA;
	
	// configuration parameters of rat head
	std::vector<float> RATHEAD_LOC;
	std::vector<float> RATHEAD_ORIENT;

	// whisking parameters
	int ACTIVE;
	std::vector<std::vector<float>> WHISKER_VEL;
	std::string file_whisking_init_angle;
	std::string file_whisking_angle;

	// head
	int SHOW_HEAD;
	std::string dir_rathead;

	// camera configuration
	btVector3 CPOS;
	float CDIST;
	float CPITCH;
	float CYAW;

	// directories/paths
	std::string dir_out;
	std::string file_video;
	std::string file_curvWall;
};

void set_default(Parameters* param);
std::vector<float> get_vector(float value, int N);

#endif //PARAMETERS_H