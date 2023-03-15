/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* 
 * This file here was adapted from the original Nvidia "boxFilterNPP.h" file 
 * available from the Cuda Sample folder. The changes here are governed
 * by the MIT license:
 *    Copyright(c) 2023 John Sogade 
 */

#include <filesystem>
#include <string>
#include <tuple>
#include<vector>
#include<thread>

#include "processImageNPP.cpp"


bool printfNPPinfo(int argc, char *argv[]) {
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}

std::tuple<std::string, std::string, std::string, int, int, int, int>
        parseCommandLineArguments(int argc, char *argv[]) {
  // (Possible) command line arguements
  std::string sFilename = "Lena.pgm";
  std::string sDirPath = "";
  std::string sResultFilename = "";
  char *filePath = NULL;
  char *output;
  int nFilterType = 1;
  int nMaskSize = 5;
  int nSrcOffset = 0;
  int nAnchor = nMaskSize / 2;

  namespace fs = std::filesystem;

  // handle input file a little differently
  if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
    getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
  } else {
    filePath = sdkFindFilePath("Lena.pgm", argv[0]);
  }

  if (filePath) {
    sFilename = filePath;
    sDirPath = fs::path(filePath).remove_filename();
  } else {
    sDirPath = fs::current_path();
    sFilename = "Lena.pgm";
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "output")) {
    getCmdLineArgumentString(argc, (const char **)argv, "output", &filePath);
    if (filePath) {
      sResultFilename = filePath;
    } else {
      sResultFilename = "";
    }
  }
  if (checkCmdLineFlag(argc, (const char **)argv, "filter")) {
    getCmdLineArgumentString(argc, (const char **)argv, "filter", &output);
    nFilterType = atoi(output);
  }
  if (checkCmdLineFlag(argc, (const char **)argv, "maskSize")) {
    getCmdLineArgumentString(argc, (const char **)argv, "maskSize", &output);
    nMaskSize = atoi(output);
    // The mask size for the the Gauss convolution filter are different from
    // the box filter -- see below
    /* Possible values:
        NPP_MASK_SIZE_1_X_3 	  => 0 - (these numbers indicates the value we 
        NPP_MASK_SIZE_1_X_5 	  => 1    set at the command line to get the  
        NPP_MASK_SIZE_3_X_1 	  => 2    corresponding gauss filter)
        NPP_MASK_SIZE_5_X_1 	  => 3
        NPP_MASK_SIZE_3_X_3 	  => 4
        NPP_MASK_SIZE_5_X_5 	  => 5
        NPP_MASK_SIZE_7_X_7 	  => 6
        NPP_MASK_SIZE_9_X_9 	  => 7
        NPP_MASK_SIZE_11_X_11 	=> 8
        NPP_MASK_SIZE_13_X_13 	=> 9
        NPP_MASK_SIZE_15_X_15 	=> 10
    */
  }
  if (checkCmdLineFlag(argc, (const char **)argv, "srcOffset")) {
    getCmdLineArgumentString(argc, (const char **)argv, "srcOffset", &output);
    nSrcOffset = atoi(output);
  }
  if (checkCmdLineFlag(argc, (const char **)argv, "anchor")) {
    getCmdLineArgumentString(argc, (const char **)argv, "anchor", &output);
    nAnchor = atoi(output);
  }

  return {sFilename, sResultFilename, sDirPath,
        nFilterType, nMaskSize, nSrcOffset, nAnchor};
}

void processImageFile(std::string sFilename,
    std::string *sResultFilename, int nFilterType,
    int nMaskSize, int nSrcOffset, int nAnchor ) {
  // if we specify the filename at the command line, then we only test
  // sFilename[0].
  int file_errors = 0;
  std::ifstream infile(sFilename.data(), std::ifstream::in);

  if (infile.good()) {
    // std::cout << "filterNPP opened: <" << sFilename.data()
    //          << "> successfully!" << std::endl;
    file_errors = 0;
    infile.close();
  } else {
    std::cout << "filterNPP unable to open: <" << sFilename.data() << ">"
              << std::endl;
    file_errors++;
    infile.close();
  }

  if (file_errors > 0) {
    exit(EXIT_FAILURE);
  }

  npp::NppRetrieveImage nppImage;
  auto [nBitDepth, sFileExt] = nppImage.ImageSetup(sFilename);

  // Do this if the output name was not provided via command line
  if (sResultFilename->compare("") == 0 || sResultFilename->empty()) {
    *sResultFilename = sFilename;

    std::string::size_type dot = sResultFilename->rfind('.');

    if (dot != std::string::npos) {
      *sResultFilename = sResultFilename->substr(0, dot);
    }

    // create output directories as needed and populate it with the
    // processed images
    namespace fs = std::filesystem;
    std::string sFilterType = "boxFilter";
    std::string szResDir = fs::path(*sResultFilename).remove_filename();
    std::string szResFile = fs::path(*sResultFilename).filename();

    if ((enumImageFilterType)nFilterType == FilterType_FilterBoxBorder) {
      sFilterType =
          FilterDescription[static_cast<int>(FilterType_FilterBoxBorder)][0];
    } else {
      sFilterType =
        FilterDescription[static_cast<int>(FilterType_FilterGaussBorder)][0];
    }
    szResDir += sFilterType + "/";
    if (fs::is_directory(szResDir) == false) {
      fs::create_directory(szResDir);
    }
    *sResultFilename = szResDir + szResFile;
    *sResultFilename += "_" + sFilterType + sFileExt;
  }

  NppProcessImage processImageNPP;
  processImageNPP.SetSrcOffset(nSrcOffset, nSrcOffset);

  if ((enumImageFilterType)nFilterType == FilterType_FilterBoxBorder) {
    // The mask size, source offset, and anchor are currently restricted
    // to both dimensions being same size
    processImageNPP.SetMaskSize(nMaskSize, nMaskSize);
    processImageNPP.SetAnchor(nAnchor, nAnchor);
    processImageNPP.SetFilterType((enumImageFilterType)nFilterType);
  } else {
    processImageNPP.SetGaussMaskSize(nMaskSize);
    processImageNPP.SetFilterType(FilterType_FilterGaussBorder);
  }

  processImageNPP.ProcessImageNPP(&nppImage, *sResultFilename, nBitDepth);
}


int main(int argc, char *argv[]) {
  printf("%s Starting...\n\n", argv[0]);

  namespace fs = std::filesystem;

  try {
    std::string sFilename = "Lena.pgm";
    std::string sResultFilename = "";
    std::string sDirPath = "";
    std::string sLogFileName = "FilterRecord.log";
    int nFilterType = 1;
    int nMaskSize = 5;
    int nSrcOffset = 0;
    int nAnchor = nMaskSize / 2;

    findCudaDevice(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false) {
      exit(EXIT_SUCCESS);
    }

    std::ofstream logFile;
    // Parse command line arguments ...
    std::tuple<std::string, std::string, std::string, int, int, int, int>
         cliArgs = parseCommandLineArguments(argc, argv);

    sFilename = std::get<0>(cliArgs);
    sResultFilename = std::get<1>(cliArgs);
    sDirPath = std::get<2>(cliArgs);
    nFilterType = std::get<3>(cliArgs);
    nMaskSize = std::get<4>(cliArgs);
    nSrcOffset = std::get<5>(cliArgs);
    nAnchor = std::get<6>(cliArgs);

    // if the filename is not * process only the single file;
    // otherwise, process all the files in the current directory
    if (fs::path(sFilename).filename().compare("*") != 0) {
      processImageFile(
          sFilename, &sResultFilename, nFilterType, nMaskSize,
          nSrcOffset, nAnchor);

      // record the event in the log file in append mode
      logFile.open(fs::path(sDirPath).parent_path().generic_string() +
                   sLogFileName, std::ios_base::app);
      logFile << "The image file, "
              << fs::path(sFilename).filename().generic_string()
              << ", was processed into " << sResultFilename
              << ", Mask (" << nMaskSize << "," << nMaskSize << ")"
              << ", Offset (" << nSrcOffset << "," << nSrcOffset << ")"
              << ", Anchor (" << nAnchor << "," << nAnchor << ")"
              << std::endl;
      logFile.close();
    } else {
      logFile.open(sDirPath + sLogFileName);

      std::vector<std::string> dirFiles;

      for (fs::directory_entry const &dEntry :
           fs::directory_iterator(sDirPath)) {
        if (dEntry.is_regular_file()) {
          std::string filepath =
          (sDirPath + dEntry.path().filename().generic_string());
          dirFiles.push_back(filepath);
        }
      }

      for (auto it = dirFiles.begin(); it != dirFiles.end(); it++) {
        std::string filepath = *it;
        processImageFile(
            filepath, &sResultFilename, nFilterType, nMaskSize,
            nSrcOffset, nAnchor);

        logFile << "The image file, "
                << fs::path(filepath).filename().generic_string()
                << ", was processed into " << sResultFilename << std::endl;
      }
      logFile.close();
    }

    exit(EXIT_SUCCESS);
  }
  catch (npp::Exception &rException) {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  }
  catch (...) {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}

