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

#include "Simulation_IO.h"

void clear_output(output *data) {

  data->Mx.clear();
  data->My.clear();
  data->Mz.clear();
  data->Fx.clear();
  data->Fy.clear();
  data->Fz.clear();
  data->T.clear();
  data->names.clear();

  for (int w = 0; w < data->Q.size(); w++) {
    data->Q[w].X.clear();
    data->Q[w].Y.clear();
    data->Q[w].Z.clear();
    data->Q[w].C.clear();
  }
}

void save_data(output *data, std::string dirname) {
  std::cout << "Saving data..." << std::endl;

  std::string subdirname = dirname + "/dynamics";
  if (!boost::filesystem::exists(subdirname)) {
    try {
      boost::filesystem::create_directories(subdirname);
    } catch (int e) {
      printf("- Error creating output directories!\n");
      exit(1);
    }
  }
  std::string filename;

  // save data to csv files
  try {
    filename = dirname + "/whisker_ID.csv";
    write_1D_string_csv(filename, data->names);
    // std::cout << "- Whisker IDs saved." << std::endl;
  } catch (...) {
    std::cout << "- Saving Whisker IDs failed." << std::endl;
  }

  // save data to csv files
  try {
    filename = subdirname + "/Mx.csv";
    write_2D_float_csv(filename, data->Mx);
    // std::cout << "- Mx saved." << std::endl;
  } catch (...) {
    std::cout << "- Saving Mx failed." << std::endl;
  }
  try {
    filename = subdirname + "/My.csv";
    write_2D_float_csv(filename, data->My);
    // std::cout << "- My saved." << std::endl;
  } catch (...) {
    std::cout << "- Saving My failed." << std::endl;
  }
  try {
    filename = subdirname + "/Mz.csv";
    write_2D_float_csv(filename, data->Mz);
    // std::cout << "- Mz saved." << std::endl;
  } catch (...) {
    std::cout << "- Saving Mz failed." << std::endl;
  }
  try {
    filename = subdirname + "/Fx.csv";
    write_2D_float_csv(filename, data->Fx);
    // std::cout << "- Fx saved." << std::endl;
  } catch (...) {
    std::cout << "- Saving Fx failed." << std::endl;
  }
  try {
    filename = subdirname + "/Fy.csv";
    write_2D_float_csv(filename, data->Fy);
    // std::cout << "- Fy saved." << std::endl;
  } catch (...) {
    std::cout << "- Saving Fy failed." << std::endl;
  }
  try {
    filename = subdirname + "/Fz.csv";
    write_2D_float_csv(filename, data->Fz);
    // std::cout << "- Fz saved." << std::endl;
  } catch (...) {
    std::cout << "- Saving Fz failed." << std::endl;
  }
}

void write_2D_float_csv(std::string filename,
                        std::vector<std::vector<float>> data) {
  std::ofstream outputFile;
  outputFile.open(filename);
  for (int row = 0; row < data.size(); row++) {
    size_t num_cols = data[row].size();
    for (int col = 0; col < num_cols - 1; col++) {
      outputFile << data[row][col] << ",";
    }
    outputFile << data[row][num_cols-1] << std::endl;
  }
  outputFile.close();
}

void write_2D_int_csv(std::string filename,
                      std::vector<std::vector<int>> data) {
  std::ofstream outputFile;
  outputFile.open(filename);
  for (int row = 0; row < data.size(); row++) {
    for (int col = 0; col < data[row].size(); col++) {
      outputFile << data[row][col] << ",";
    }
    outputFile << std::endl;
  }
  outputFile.close();
}

void write_1D_string_csv(std::string filename, std::vector<std::string> data) {
  std::ofstream outputFile;
  outputFile.open(filename);
  for (int row = 0; row < data.size(); row++) {
    outputFile << data[row] << std::endl;
  }
  outputFile.close();
}

void read_csv_string(std::string fileName, std::vector<std::string> &dataList) {

  std::ifstream file(fileName);
  std::string line = "";
  std::string delimeter = ",";

  if (file.good()) {
    // Iterate through each line and split the content using delimeter
    while (getline(file, line)) {

      dataList.push_back(line);
    }
    // Close the File
    file.close();
  } else {
    std::cout << "\n======== ABORT SIMULATION ========" << std::endl;
    std::cout << "Failure in loading file " << fileName << "\n" << std::endl;
    exit(EXIT_FAILURE);
  }
}

void read_csv_int(std::string fileName,
                  std::vector<std::vector<int>> &dataList) {

  std::ifstream file(fileName);
  std::string line = "";
  std::string delimeter = ",";

  if (file.good()) {
    // Iterate through each line and split the content using delimeter
    while (getline(file, line)) {
      std::vector<std::string> vec_string;
      std::vector<int> vec_num;
      boost::algorithm::split(vec_string, line, boost::is_any_of(delimeter));

      // convert to float
      for (int i = 0; i < vec_string.size(); i++) {
        vec_num.push_back(boost::lexical_cast<int>(vec_string[i]));
      }

      dataList.push_back(vec_num);
    }
    // Close the File
    file.close();
  } else {
    std::cout << "\n======== ABORT SIMULATION ========" << std::endl;
    std::cout << "Failure in loading file " << fileName << "\n" << std::endl;
    exit(EXIT_FAILURE);
  }
}

void read_csv_float(std::string fileName,
                    std::vector<std::vector<float>> &dataList) {
  std::ifstream file(fileName);
  std::string line = "";
  std::string delimeter = ",";

  if (file.good()) {
    // Iterate through each line and split the content using delimeter
    while (getline(file, line)) {
      std::vector<std::string> vec_string;
      std::vector<float> vec_num;
      boost::algorithm::split(vec_string, line, boost::is_any_of(delimeter));

      // convert to float
      for (int i = 0; i < vec_string.size(); i++) {
        vec_num.push_back(boost::lexical_cast<float>(vec_string[i]));
      }

      dataList.push_back(vec_num);
    }
    // Close the File
    file.close();
  } else {
    std::cout << "\n======== ABORT SIMULATION ========" << std::endl;
    std::cout << "Failure in loading file " << fileName << "\n" << std::endl;
    exit(EXIT_FAILURE);
  }
}
