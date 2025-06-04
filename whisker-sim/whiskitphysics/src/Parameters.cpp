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

#include "Parameters.h"
#include <string>

// set default parameter values
Parameters::Parameters() {

  // input arguments for simulator
  NUM_RATS = 1;    // set number of rats
  DEBUG = 0;       // enable debug mode
  HZ = 100;        // set simulation frequency (frames per second)
  SUBSTEPS = 100;  // set physics substep calculation per simulation frame
  TIME_STOP = 1.0; // set overall simulation time
  PRINT = 0; // set to PRINT=1 to kinematics/dynamics realtime, set to PRINT = 2
             // to print simulation time
  SAVE = 0;  // save results to csv file
  SAVE_VIDEO = 0; // save video when running main_opengl.cpp

  // collision object type
  OBJ_CONVEX_HULL = 0; // set to 1 to load object with convex hull decomposition
  OBJ_PATH = "";
  OBJ_SCALE = 1.0;
  OBJ_SPEED = 1.0;
  OBJ_POS = btVector3(5, 30, 5);
  OBJ_ORI = btVector3(0, 0, 0);
  OBJ_THETA = 0;
  OBJ_MASS = 1.0; // 0 mass makes it a static body

  // specify whisker configuration parameters
  WHISKER_MODEL_DIR = "whisker_param_average_mouse";
  WHISKER_NAMES = {"RA0","RA1","RA2","RA3","RA4",
                   "RB0","RB1","RB2","RB3","RB4",
                   "RC0","RC1","RC2","RC3","RC4","RC5","RC6",
                   "RD0","RD1","RD2","RD3","RD4","RD5","RD6",
                   "RE0","RE1","RE2","RE3","RE4","RE5"};

  BLOW = 1; // increase whisker diameter for better visualization (will affect
            // dynamics!!)
  NO_CURVATURE = 0;      // disable curvature
  NO_MASS = 0;           // disable mass of bodies
  NO_WHISKERS = 0;       // disable whiskers
  NUM_LINKS_PER_MM = 1.0;  // set number of links per mm
  RHO_BASE = 1260.0 / 2; // set density at whisker base
  RHO_TIP = 1690.0 / 2;  // set density at whisker tip
  E = 5e9;               // set young's modulus (GPa) at whisker base
  ZETA = 0.32;           // set damping coefficient zeta at whisker base

  // enable/disable whisking mode for added whiskers
  // Note: the whisking trajectory is pre-specified by user.
  ACTIVE = 0;
  file_whisking_init_angle =
      ACTIVE ? "whisking_init_angle.csv" : "param_bp_angles.csv";
  file_whisking_angle = "whisking_trajectory.csv";

  // paths
  std::string sourceFilePath = __FILE__;
  boost::filesystem::path sourcePath(sourceFilePath);
  boost::filesystem::path parentDir = sourcePath.parent_path();
  boost::filesystem::path ratHeadPath = parentDir / "../data/object/NewRatHead.obj";
  boost::filesystem::path whiskerPath = parentDir / ("../data/" + WHISKER_MODEL_DIR + "/");
  dir_param = whiskerPath.string();

  // enable/disable rat head
  SHOW_HEAD = 1;
  dir_rathead = SHOW_HEAD ? ratHeadPath.string() : "";

  // rat position/orientation parameters
  RATHEAD_LOC = {0, 0, 0};    // set position of rathead
  RATHEAD_ORIENT = {0, 0, 0}; // set initial heading of rathead

  // camera parameters for visualization
  CPOS = btVector3(0, 0, 4); // set camera pos
  CDIST = 40;                // set camera distance
  CPITCH = -89;              // set camera pitch
  CYAW = 0;                  // set camera yaw

  // input/output file paths
  dir_out = "../output/test";
  file_video = "video.mp4"; // just the file name
}

// create a vector with same value
std::vector<float> get_vector(float value, int N) {
  std::vector<float> vect;
  for (int i = 0; i < N; i++) {
    vect.push_back(value);
  }
  return vect;
}