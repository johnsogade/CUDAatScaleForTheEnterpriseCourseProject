
/* Note some parts contains code was adpated from Nvidia's CUDA sample projects.
 * These parts are subject to the Nvidia and other third party license and 
 * copyright information contained in those files. Otherwise all other parts of
 * this project follow the  MIT License:
 *
 *    Copyright(c) 2023 John Sogade
 *
 *    Permission is hereby granted,
 *    free of charge, to any person obtaining a copy of this software and
 *    associated documentation files(the "Software"), to deal in the Software
 *    without restriction, including without limitation the rights to use, 
 *    copy, modify, merge, publish, distribute, sublicense, and / or sell 
 *    copies of the Software, and to permit persons to whom the Software is
 *    furnished to do so, subject to the following conditions :
 *
 *    The above copyright notice and this permission notice shall be included
 *    in all copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS",
 *   WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
 *   TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 *   DAMAGES OR OTHER
 *   LIABILITY,
 *   WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
 *   THE SOFTWARE.
 */

#ifndef SRC_PROCESSIMAGENPP_H_
#define SRC_PROCESSIMAGENPP_H_

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include "ImageIOEx.h"
#include <ImagesCPU.h>

#include <ImagesNPP.h>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>
#include <string.h>

#include <string>
#include <fstream>
#include <iostream>
#include <vector>

    enum enumImageFilterType {
        FilterType_FilterUnused = 0,
        FilterType_FilterBoxBorder = 1,
        FilterType_FilterGaussBorder = 2,
        FilterType_Unsupported = 3
    };

const std::vector<std::vector<std::string>> FilterDescription = {
    {static_cast<int>(FilterType_FilterUnused), "unused"},
    {static_cast<int>(FilterType_FilterBoxBorder), "boxFilter"},
    {static_cast<int>(FilterType_FilterGaussBorder), "gaussFilter"},
    {static_cast<int>(FilterType_Unsupported), "unsupported"}};

class NppProcessImage {
    enumImageFilterType nFilterType = FilterType_FilterBoxBorder;

    // create structs with box-filter mask and source offset size
    NppiSize oMaskSize = {5, 5};
    NppiPoint oSrcOffset = {0, 0};
    // set anchor point inside the mask to (oMaskSize.width / 2,
    // oMaskSize.height / 2) It should round down when odd
    NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};

    NppiMaskSize oGaussMaskSize = NPP_MASK_SIZE_5_X_5;
    /* Possible values:
        NPP_MASK_SIZE_1_X_3 	
        NPP_MASK_SIZE_1_X_5 	
        NPP_MASK_SIZE_3_X_1 	
        NPP_MASK_SIZE_5_X_1 	
        NPP_MASK_SIZE_3_X_3 	
        NPP_MASK_SIZE_5_X_5 	
        NPP_MASK_SIZE_7_X_7 	
        NPP_MASK_SIZE_9_X_9 	
        NPP_MASK_SIZE_11_X_11 	
        NPP_MASK_SIZE_13_X_13 	
        NPP_MASK_SIZE_15_X_15 	
    */

    void ProcessC1Image(
        npp::NppRetrieveImage *pImageSetter,
        std::string sResultFilename,
        int nBitDepth);
    void ProcessC2Image(npp::NppRetrieveImage *pImageSetter,
                    std::string sResultFilename,
                    int nBitDepth);
    void ProcessC3Image(npp::NppRetrieveImage *pImageSetter,
                    std::string sResultFilename,
                    int nBitDepth);
    void ProcessC4Image(npp::NppRetrieveImage *pImageSetter,
                    std::string sResultFilename,
                    int nBitDepth);

 public:
    void SetMaskSize(int width, int height);
    void SetSrcOffset(int x, int y);
    void SetAnchor(int x, int y);
    void SetGaussMaskSize(int nMaskSize);
    void SetFilterType(enumImageFilterType nType);
    void ProcessImageNPP(npp::NppRetrieveImage *pImageSetter,
                     std::string szResultFileName,
                     int nBitDepth);
};
#endif  //  SRC_PROCESSIMAGENPP_H_
