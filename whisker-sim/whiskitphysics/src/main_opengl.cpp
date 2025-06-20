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

#include "CommonInterfaces/CommonCameraInterface.h"
#include "OpenGLWindow/OrthographicCamera.h"
#include "Simulation.h"

#include "CommonInterfaces/CommonExampleInterface.h"
#include "CommonInterfaces/CommonGUIHelperInterface.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "BulletCollision/CollisionShapes/btCollisionShape.h"
#include "BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h"
#include "OpenGLWindow/SimpleOpenGL3App.h"
#include "OpenGLWindow/OrthographicCamera.h"
#include "Bullet3Common/b3Quaternion.h"

#include "Utils/b3Clock.h"

#include <stdio.h>
#include "ExampleBrowser/OpenGLGuiHelper.h"

#include <iostream>
// #include <vector>
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

char* gVideoFileName = 0;
char* gPngFileName = 0;

float gWidth = 1024;
float gHeight = 768;

SimpleOpenGL3App* app;
OrthographicCamera* ortho_cam = new OrthographicCamera();

CommonCameraInterface* default_cam;

Simulation* simulation;
int gSharedMemoryKey=-1;


static void	OnKeyboard(int key, int state) {
	// std::cout << "keyboard pressed " << key << " state " << state << std::endl;
	if (state == 1) {
		switch (key) {
			case 'q': {
				exitFlag = 1;
				break;
			}
			case '1': {
				// custom/default view
				simulation->resetCamera();
				break;
			}
			case '2': {
				// top view
				simulation->setCamera(15, 90, -89, 10, 0, 10);
				break;
			}
			case '3': {
				// front right 45 side view
				simulation->setCamera(15, 180, -45, 5, 10, 5);
				break;
			}
			case '4': {
				// front view
				simulation->setCamera(10, -10, -90, 10, 0, 10);
				break;
			}
			case 'o': {
				// orthogonal projection
				ortho_cam->setOrthoBounds(-30, 30, -30, 30);
				app->m_instancingRenderer->setActiveCamera(ortho_cam);
				simulation->resetCamera();
				break;
			}
			case 'p': {
				// perspective projection
				app->m_instancingRenderer->setActiveCamera(default_cam);
				break;
			}
		
		}
	}
}

class LessDummyGuiHelper : public DummyGUIHelper
{
	CommonGraphicsApp* m_app;
public:
	virtual CommonGraphicsApp* getAppInterface()
	{
		return m_app;
	}

	LessDummyGuiHelper(CommonGraphicsApp* app)
		:m_app(app)
	{
	}
};

int main(int argc, char** argv) 
{ 
	signal(SIGINT, exit_function);
	Parameters* param = new Parameters();
	
	

  	try 
  	{ 
    /** Define and parse the program options 
     */ 
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
		("SAVE_VIDEO", po::value<int>(&param->SAVE_VIDEO), "video on/off")

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
		("SHOW_HEAD", po::value<int>(&param->SHOW_HEAD), "show head on/off")
		
		("POSITION", po::value<std::vector<std::string> >()->multitoken(), "initial position of rat")
		("ORIENTATION", po::value<std::vector<std::string> >()->multitoken(), "initial orientation of rat (euler angles)")

		("CX", po::value<float>(&param->CPOS[0]), "camera x pos")
		("CY", po::value<float>(&param->CPOS[1]), "camera y pos")
		("CZ", po::value<float>(&param->CPOS[2]), "camera z pos")
		("CDIST", po::value<float>(&param->CDIST), "camera distance")
		("CPITCH", po::value<std::string>(), "head pitch")
		("CYAW", po::value<std::string>(), "head yaw")
		
		("dir_out", po::value<std::string>(&param->dir_out), "foldername for output file")
		("file_video", po::value<std::string>(&param->file_video), "filename of video");


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

			if (vm.count("CPITCH")){
				std::string cpitch;
				cpitch = vm["CPITCH"].as<std::string>();
				param->CPITCH = lexical_cast<float>(cpitch);
			}
			
			if (vm.count("CYAW")){
				std::string cyaw;
				cyaw = vm["CYAW"].as<std::string>();
				param->CYAW = lexical_cast<float>(cyaw);
			}

	    	
			app = new SimpleOpenGL3App("Bullet Whisker Simulation",1024,768,true);
			default_cam = app->m_instancingRenderer->getActiveCamera();
			// use ortho cam on load
			ortho_cam->setOrthoBounds(-30, 30, -30, 30);
			app->m_instancingRenderer->setActiveCamera(ortho_cam);
			// simulation->resetCamera();
			
			app->m_window->setKeyboardCallback((b3KeyboardCallback)OnKeyboard);
			
			OpenGLGuiHelper gui(app,false);
			CommonExampleOptions options(&gui);
			

			simulation = new Simulation(options.m_guiHelper);
			simulation->processCommandLineArgs(argc, argv);

			// save parameters in simulation object
			simulation->parameters = param;
			simulation->initPhysics();
			simulation->resetCamera();

			char fileName[1024];
			int textureWidth = 128;
			int textureHeight = 128;
	
			unsigned char*	image = new unsigned char[textureWidth*textureHeight * 4];
			int textureHandle = app->m_renderer->registerTexture(image, textureWidth, textureHeight);

			if(param->SAVE_VIDEO){
				std::string videoname = param->dir_out + "/" + param->file_video;
				gVideoFileName = &videoname[0];
				
				if (gVideoFileName){
					if (!boost::filesystem::exists(param->dir_out)) {
						try {
						  boost::filesystem::create_directories(param->dir_out);
						} catch (int e) {
						  printf("- Error creating output directories!\n");
						  exit(1);
						}
					}

					std::cout << "Rendering video..." << std::endl;
					app->dumpFramesToVideo(gVideoFileName);
	
				}

				std::string pngname = "png_test";
				gPngFileName = &pngname[0];
				
				app->m_renderer->writeTransforms();
			}
			
			do
			{
				if(param->SAVE_VIDEO){
					static int frameCount = 0;
					frameCount++;

					if (gPngFileName)
					{
						sprintf(fileName, "%s%d.png", gPngFileName, frameCount++);
						app->dumpNextFrameToPng(fileName);
					}
		
					//update the texels of the texture using a simple pattern, animated using frame index
					for (int y = 0; y < textureHeight; ++y)
					{
						const int	t = (y + frameCount) >> 4;
						unsigned char*	pi = image + y*textureWidth * 3;
						for (int x = 0; x < textureWidth; ++x)
						{
							const int		s = x >> 4;
							const unsigned char	b = 180;
							unsigned char			c = b + ((s + (t & 1)) & 1)*(255 - b);
							pi[0] = pi[1] = pi[2] = pi[3] = c; pi += 3;
						}
					}
		
					app->m_renderer->activateTexture(textureHandle);
					app->m_renderer->updateTexture(textureHandle, image);
		
					float color[4] = { 255, 1, 1, 1 };
					app->m_primRenderer->drawTexturedRect(100, 200, gWidth / 2 - 50, gHeight / 2 - 50, color, 0, 0, 1, 1, true);
				}

				app->m_instancingRenderer->init();
		    	app->m_instancingRenderer->updateCamera(app->getUpAxis());
			
				simulation->stepSimulation();

				if(param->DEBUG!=1){
					simulation->renderScene();
					app->m_renderer->renderScene();
				}
				

				// DrawGridData dg;
		        // dg.upAxis = app->getUpAxis();
				app->setBackgroundColor(1,1,1);
				// app->drawGrid(dg);
				app->swapBuffer();


			} while (!app->m_window->requestedExit() && !(exitFlag || simulation->exitSim));
			
			
			if(simulation->parameters->SAVE){
				std::cout << "Simulation terminated." << std::endl;
				std::cout << "Saving data..." << std::endl;
				output* results = simulation->get_results();
				save_data(results,simulation->parameters->dir_out);
			}

			std::cout << "Exit simulation..." << std::endl;
			simulation->exitPhysics();
			delete simulation;
			delete app;
			delete[] image;
			std::cout << "Done." << std::endl;

	    } 
	    catch(po::error& e) 
	    { 
	      std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
	      std::cerr << desc << std::endl; 
	      return 1; 
	    } 
 
    // application code here // 
 
  	} 
  	catch(std::exception& e) 
  	{ 
    	std::cerr << "Unhandled Exception reached the top of main: " 
              << e.what() << ", application will now exit" << std::endl; 
    	return 2; 
 
  	} 
 
  	return 0; 
 
} // main 



