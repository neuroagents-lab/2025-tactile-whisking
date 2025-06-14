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

This code is based on code published by
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2015 Google Inc. http://bulletphysics.org

*/

#include "Simulation.h"
#include "CommonInterfaces/CommonExampleInterface.h"
#include "CommonInterfaces/CommonGUIHelperInterface.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "BulletCollision/CollisionShapes/btCollisionShape.h"
#include "BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h"
#include "OpenGLWindow/SimpleOpenGL3App.h"
#include "Bullet3Common/b3Quaternion.h"

#include "LinearMath/btTransform.h"
#include "LinearMath/btHashMap.h"

#include <iostream>
#include <boost/program_options.hpp>

#include <signal.h>
#include <stdlib.h>
#include <string>

#include <boost/lexical_cast.hpp>
using boost::lexical_cast;

volatile sig_atomic_t exitFlag = 0;

void exit_function(int sigint)
{
	exitFlag = 1;
}

int main(int argc, char* argv[])
{
	signal(SIGINT, exit_function);
	Parameters* param = new Parameters();

  	try 
  	{ /** Define and parse the program options  */ 
	    namespace po = boost::program_options; 
	    po::options_description desc("Options");
	    desc.add_options() 
		("help,h", "Help screen")
		("DEBUG", po::value<int>(&param->DEBUG), "debug on/off")
		
		("TIME_STOP", po::value<float>(&param->TIME_STOP), "duration of simulation")
		("HZ", po::value<float>(&param->HZ), "frames (num of physics steps) per second")
		("SUBSTEPS", po::value<int>(&param->SUBSTEPS), "num of physics substeps per simulation frame")

		("PRINT", po::value<int>(&param->PRINT), "print simulation output")
		("SAVE", po::value<int>(&param->SAVE), "saving on/off")

		("OBJ_CONVEX_HULL", po::value<int>(&param->OBJ_CONVEX_HULL), "load object with convex hull decomposition on/off")
		("OBJ_PATH", po::value<std::string>(&param->OBJ_PATH), "path to object .obj")
		("OBJ_SCALE", po::value<float>(&param->OBJ_SCALE), "scalar multiplier for scale")
		("OBJ_SPEED", po::value<float>(&param->OBJ_SPEED), "scalar multiplier for speed")
		("OBJ_X", po::value<float>(&param->OBJ_POS[0]), "object x distance from origin from closest point in object")
		("OBJ_Y", po::value<float>(&param->OBJ_POS[1]), "object y pos")
		("OBJ_Z", po::value<float>(&param->OBJ_POS[2]), "object z pos")
		("OBJ_THETA", po::value<float>(&param->OBJ_THETA), "object rotation about z axis in degrees")
		
		("WHISKER_MODEL_DIR", po::value<std::string>(&param->WHISKER_MODEL_DIR), "data/whisker_param folder to use, e.g. whisker_param_average_mouse")
		("WHISKER_NAMES", po::value<std::vector<std::string> >(&param->WHISKER_NAMES)->multitoken(), "whisker names to simulate")
		("WHISKER_LINKS_PER_MM", po::value<float>(&param->NUM_LINKS_PER_MM), "number of links per mm for each whisker")

		("ACTIVE", po::value<int>(&param->ACTIVE), "active on/off")
		
		("POSITION", po::value<std::vector<std::string> >()->multitoken(), "initial position of rat")
		("ORIENTATION", po::value<std::vector<std::string> >()->multitoken(), "initial orientation of rat (euler angles)")
		
		("CX", po::value<float>(&param->CPOS[0]), "camera x pos")
		("CY", po::value<float>(&param->CPOS[1]), "camera y pos")
		("CZ", po::value<float>(&param->CPOS[2]), "camera z pos")
		("CDIST", po::value<float>(&param->CDIST), "camera distance")
		("CPITCH", po::value<std::string>(), "head pitch")
		("CYAW", po::value<std::string>(), "head yaw")
		
		("dir_out", po::value<std::string>(&param->dir_out), "foldername for output file");


	    po::variables_map vm; 


	    try { 
		    po::store(po::parse_command_line(argc, argv, desc,po::command_line_style::unix_style ^ po::command_line_style::allow_short), vm); // can throw 
		 	po::notify(vm);

		 	if ( vm.count("help")  ) { 
		        std::cout << "Bullet Whisker Simulation" << std::endl 
		                  << desc << std::endl; 
		        return 0; 
		    } 
			
			if (param->WHISKER_NAMES[0] == "ALL"){
	    		param->WHISKER_NAMES = {
	    			"LA0","LA1","LA2","LA3","LA4",
	    			"LB0","LB1","LB2","LB3","LB4",
	    			"LC0","LC1","LC2","LC3","LC4","LC5","LC6",
	    			"LD0","LD1","LD2","LD3","LD4","LD5","LD6",
	    			"LE0","LE1","LE2","LE3","LE4","LE5",
	    			"RA0","RA1","RA2","RA3","RA4",
	    			"RB0","RB1","RB2","RB3","RB4",
	    			"RC0","RC1","RC2","RC3","RC4","RC5","RC6",
	    			"RD0","RD1","RD2","RD3","RD4","RD5","RD6",
	    			"RE0","RE1","RE2","RE3","RE4","RE5"};
	    	}
	    	else if (param->WHISKER_NAMES[0] == "C"){
	    		param->WHISKER_NAMES = { "RC0","RC1","RC2","RC3" };
	    	}
	    	else if (param->WHISKER_NAMES[0] == "R"){
	    		param->WHISKER_NAMES = {
	    			"RA0","RA1","RA2","RA3","RA4",
	    			"RB0","RB1","RB2","RB3","RB4",
	    			"RC0","RC1","RC2","RC3","RC4","RC5","RC6",
	    			"RD0","RD1","RD2","RD3","RD4","RD5","RD6",
	    			"RE0","RE1","RE2","RE3","RE4","RE5"};
	    	}
			else if (param->WHISKER_NAMES[0] == "L"){
	    		param->WHISKER_NAMES = {
	    			"LA0","LA1","LA2","LA3","LA4",
	    			"LB0","LB1","LB2","LB3","LB4",
	    			"LC0","LC1","LC2","LC3","LC4","LC5",
	    			"LD0","LD1","LD2","LD3","LD4","LD5",
	    			"LE1","LE2","LE3","LE4","LE5"};
	    	}

			std::vector<std::string> coordinates;
			if (!vm["POSITION"].empty() && (coordinates = vm["POSITION"].as<std::vector<std::string> >()).size() == 3) {
				param->RATHEAD_LOC[0] = lexical_cast<float>(coordinates[0]);
				param->RATHEAD_LOC[1] = lexical_cast<float>(coordinates[1]);
				param->RATHEAD_LOC[2] = lexical_cast<float>(coordinates[2]);
			}	

			std::vector<std::string> angles;
			if (!vm["ORIENTATION"].empty() && (angles = vm["ORIENTATION"].as<std::vector<std::string> >()).size() == 3) {
				param->RATHEAD_ORIENT[0] = lexical_cast<float>(angles[0]) / 180 * PI;
				param->RATHEAD_ORIENT[1] = lexical_cast<float>(angles[1]) / 180 * PI;
				param->RATHEAD_ORIENT[2] = lexical_cast<float>(angles[2]) / 180 * PI;
			}		

			
		  	DummyGUIHelper noGfx;

			CommonExampleOptions options(&noGfx);
			Simulation* simulation = new Simulation(options.m_guiHelper);

			// save parameters in simulation object
			simulation->parameters = param;
			simulation->initPhysics();
			std::cout.precision(17);
			
			// run simulation
			do{
				simulation->stepSimulation();
			}while(!(exitFlag || simulation->exitSim) );

			std::cout << "Saving data..." << std::endl;
			if(simulation->parameters->SAVE){
				std::cout << "Simulation terminated." << std::endl;
				output* results = simulation->get_results();
				save_data(results,simulation->parameters->dir_out);
			}
			

			std::cout << "Exit simulation..." << std::endl;
			simulation->exitPhysics();

			delete simulation;
			std::cout << "Done." << std::endl;
			
	    } 
	    catch(po::error& e) 
	    { 
	      std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
	      std::cerr << desc << std::endl; 
	      return 1; 
	    } 
 
  	} 
  	catch(std::exception& e) 
  	{ 	std::cerr << "Unhandled Exception reached the top of main: "           << e.what() << ", application will now exit" << std::endl; 	return 2; 
 
  	} 

	
	return 0;
}
