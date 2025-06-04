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

btVector4 BLUE = btVector4(0., 0., 1.0, 1);
btVector4 RED = btVector4(1., 0., 0.0, 1);
btVector4 GREEN = btVector4(0., 1., 0.0, 1);
btVector4 GREY = btVector4(0., 0., 0.0, 0.5);
btVector4 YELLOW = btVector4(1., 1., 0.0, 1);
btVector4 ORANGE = btVector4(1., 0.647, 0.0, 1);

void Simulation::stepSimulation() {
  auto start = std::chrono::high_resolution_clock::now();
  m_time +=  1 / parameters->HZ; // increase time
  m_step += 1;                   // increase step

  // run simulation as long as stop time not exceeded
  if (parameters->TIME_STOP == 0 || m_time < parameters->TIME_STOP) {

    for (Rat *rat : rats) {
      rat->detect_collision(m_dynamicsWorld);
      if (!parameters->NO_WHISKERS && parameters->SAVE) {
        rat->dump_M(data_dump);
        rat->dump_F(data_dump);
        rat->dump_Q(data_dump);
      }
      if (parameters->ACTIVE) {
			  rat->whisk(m_step, parameters->WHISKER_VEL);
      }
    }
    

    // step simulation
    m_dynamicsWorld->stepSimulation(
        1/parameters->HZ, parameters->SUBSTEPS,
        1/parameters->HZ / parameters->SUBSTEPS);

    // draw debug if enabled
    if (parameters->DEBUG) {
      m_dynamicsWorld->debugDrawWorld();
    }

    // set exit flag to zero
    exitSim = 0;
  } else {
    // timeout -> set exit flg
    exitSim = 1;
  }

  if (parameters->PRINT == 2) {
    std::cout << "Simulation time: " << std::setprecision(2) << m_time << "s" << std::endl;
  }
}

void Simulation::initPhysics() {
  vec = btVector3(0.5, -1, 0).normalized();
  data_dump->init(parameters->WHISKER_NAMES);

  // set visual axis
  m_guiHelper->setUpAxis(2);

  // create empty dynamics world[0]
  m_collisionConfiguration = new btDefaultCollisionConfiguration();
  m_dispatcher = new btCollisionDispatcher(m_collisionConfiguration);

  // broadphase algorithm
  m_broadphase = new btDbvtBroadphase();

  // select solver
  m_solver = new btSequentialImpulseConstraintSolver();

  m_dynamicsWorld = new btDiscreteDynamicsWorld(
      m_dispatcher, m_broadphase, m_solver, m_collisionConfiguration);

  // set number of iterations
  m_dynamicsWorld->getSolverInfo().m_numIterations = 20;
  m_dynamicsWorld->getSolverInfo().m_solverMode =
      SOLVER_SIMD | SOLVER_USE_WARMSTARTING | SOLVER_RANDMIZE_ORDER | 0;
  m_dynamicsWorld->getSolverInfo().m_splitImpulse = true;
  m_dynamicsWorld->getSolverInfo().m_erp = 0.8f;

  // set gravity
  m_dynamicsWorld->setGravity(btVector3(0, 0, 0));

  // create debug drawer
  m_guiHelper->createPhysicsDebugDrawer(m_dynamicsWorld);

  if (m_dynamicsWorld->getDebugDrawer()) {
    if (parameters->DEBUG == 1) {
      std::cout << "DEBUG option 1: wireframes." << std::endl;
      m_dynamicsWorld->getDebugDrawer()->setDebugMode(
          btIDebugDraw::DBG_DrawWireframe);
    } else if (parameters->DEBUG == 2) {
      std::cout << "DEBUG option 2: constraints." << std::endl;
      m_dynamicsWorld->getDebugDrawer()->setDebugMode(
          btIDebugDraw::DBG_DrawConstraints);
    } else if (parameters->DEBUG == 3) {
      std::cout << "DEBUG option 3: wireframes & constraints." << std::endl;
      m_dynamicsWorld->getDebugDrawer()->setDebugMode(
          btIDebugDraw::DBG_DrawWireframe +
          btIDebugDraw::DBG_DrawConstraintLimits);
    } else if (parameters->DEBUG == 4) {
      std::cout << "DEBUG option 4: AAbb." << std::endl;
      m_dynamicsWorld->getDebugDrawer()->setDebugMode(
          btIDebugDraw::DBG_DrawAabb);
    } else if (parameters->DEBUG == 5) {
      std::cout << "DEBUG option 5: Frammes." << std::endl;
      m_dynamicsWorld->getDebugDrawer()->setDebugMode(
          btIDebugDraw::DBG_DrawFrames);
    } else if (parameters->DEBUG == 6) {
      std::cout << "DEBUG option 6: Only collision" << std::endl;
      // m_dynamicsWorld->getDebugDrawer()->setDebugMode(btIDebugDraw::DBG_DrawFrames);
    } else {
      std::cout << "No DEBUG." << std::endl;
      m_dynamicsWorld->getDebugDrawer()->setDebugMode(
          btIDebugDraw::DBG_NoDebug);
    }
  }

  // rats!!
  for (int i = 0; i < parameters->NUM_RATS; ++i) {
    btTransform tf_i = createFrame(btVector3(30*i, 0, 0), btVector3(0, 0, 0));
    btTransform tf_origin = createFrame(btVector3(0, 0, 0), btVector3(0, 0, 0));

    Rat* rat = new Rat(m_guiHelper, m_dynamicsWorld, &m_collisionShapes, parameters, tf_i);
    rats.push_back(rat);

    // create object to collide with for each rat
    btVector4 objectColor = btVector4(0.6, 0.6, 1.0, 1);
    Object* obj = new Object(
      m_guiHelper, m_dynamicsWorld, &m_collisionShapes, tf_origin,
      parameters->OBJ_PATH, objectColor, btScalar(parameters->OBJ_SCALE),
      btScalar(parameters->OBJ_MASS), COL_ENV, envCollidesWith, parameters->OBJ_CONVEX_HULL);
    // adjust object position so that its -x side is X away from origin
    obj->setOrientation(btVector3(0, 0, 1), parameters->OBJ_THETA * PI / 180);
    btVector3 adjusted_pos = parameters->OBJ_POS - btVector3(obj->xyz_min[0], -obj->xyz_min[1], 0) + tf_i.getOrigin();
    obj->setPosition(adjusted_pos);
    obj->body->setLinearVelocity(btVector3(0, -1, 0) * parameters->OBJ_SPEED);
    objects.push_back(obj);
  }

  // generate graphics
  m_guiHelper->autogenerateGraphicsObjects(m_dynamicsWorld);

  // set camera position
  camPos[0] = parameters->CPOS[0];
  camPos[1] = parameters->CPOS[1];
  camPos[2] = parameters->CPOS[2];
  camDist = parameters->CDIST;
  camPitch = parameters->CPITCH;
  camYaw = parameters->CYAW;
  resetCamera();

	// if active whisking, load whisking protraction angle trajectory
	if (parameters->ACTIVE){
		read_csv_float(parameters->dir_param + parameters->file_whisking_angle, parameters->WHISKER_VEL);
		parameters->TIME_STOP = std::min(parameters->TIME_STOP, (parameters->WHISKER_VEL[0].size()/3 - 1) / parameters->HZ);
	}

  // initialize time/step tracker
  m_time_elapsed = 0;
  m_time = 0;
  m_step = 0;

  if (parameters->PRINT > 0) {
    std::cout << "Start simulation..." << std::endl;
    std::cout << "====================================================" << std::endl;
  }
}

output *Simulation::get_results() { return data_dump; }

void Simulation::resetCamera() {
  setCamera(camDist, camYaw, camPitch, camPos[0], camPos[1], camPos[2]);
}

void Simulation::setCamera(float dist, float yaw, float pitch, float x, float y,
                           float z) {
  m_guiHelper->resetCamera(dist, yaw, pitch, x, y, z);
}