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

#include "processImageNPP.h"
#include <string>


void NppProcessImage::ProcessC1Image(npp::NppRetrieveImage *pImageSetter,
                             std::string sResultFilename, int nBitDepth) {
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image from disk
    pImageSetter->loadImage(&oHostSrc, nBitDepth);
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);
    NppiSize oSrcSize = {static_cast<int>(oDeviceSrc.width()),
                        static_cast<int>(oDeviceSrc.height())};
    // create struct with ROI size
    NppiSize oSizeROI = {static_cast<int>(oDeviceSrc.width()),
                        static_cast<int>(oDeviceSrc.height())};

    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
    if (nFilterType == FilterType_FilterBoxBorder) {
        // run box filter
        NPP_CHECK_NPP(nppiFilterBoxBorder_8u_C1R(
            oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
            oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, oMaskSize, oAnchor,
            NPP_BORDER_REPLICATE));
    } else if (nFilterType == FilterType_FilterGaussBorder) {
        Npp32s nSrcStep = oDeviceSrc.pitch();
        Npp32s nDstStep = oDeviceDst.pitch();

        // run gauss border filter
        NPP_CHECK_NPP(nppiFilterGaussBorder_8u_C1R(
            oDeviceSrc.data(), nSrcStep, oSrcSize, oSrcOffset,
            oDeviceDst.data(), nDstStep, oSizeROI, oGaussMaskSize,
            NPP_BORDER_REPLICATE));
    }

    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
    // save host image to result file
    pImageSetter->saveImage(sResultFilename, oHostDst.data(),
                    oHostDst.pitch(), oHostDst.height(), oHostDst.width());
    // std::cout << "Saved image: " << sResultFilename << std::endl;

    cudaDeviceSynchronize();

    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());
}

void NppProcessImage::ProcessC2Image(npp::NppRetrieveImage *pImageSetter,
                             std::string sResultFilename, int nBitDepth) {
    // Not implemented
}

void NppProcessImage::ProcessC3Image(npp::NppRetrieveImage *pImageSetter,
                              std::string sResultFilename, int nBitDepth) {
    npp::ImageCPU_8u_C3 oHostSrc;
    // load color image from disk
    pImageSetter->loadImage(&oHostSrc, nBitDepth);
    npp::ImageNPP_8u_C3 oDeviceSrc(oHostSrc);
    NppiSize oSrcSize = {static_cast<int>(oDeviceSrc.width()),
                        static_cast<int>(oDeviceSrc.height())};
    // create struct with ROI size
    NppiSize oSizeROI = {static_cast<int>(oDeviceSrc.width()),
                        static_cast<int>(oDeviceSrc.height())};
    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C3 oDeviceDst(oSizeROI.width, oSizeROI.height);
    if (nFilterType == FilterType_FilterBoxBorder) {
        // run box filter
        NPP_CHECK_NPP(nppiFilterBoxBorder_8u_C3R(
            oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
            oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, oMaskSize, oAnchor,
            NPP_BORDER_REPLICATE));
    } else if (nFilterType == FilterType_FilterGaussBorder) {
        Npp32s nSrcStep = oDeviceSrc.pitch();
        Npp32s nDstStep = oDeviceDst.pitch();

        // run gauss border filter
        NPP_CHECK_NPP(nppiFilterGaussBorder_8u_C3R(
            oDeviceSrc.data(), nSrcStep, oSrcSize, oSrcOffset,
            oDeviceDst.data(), nDstStep, oSizeROI, oGaussMaskSize,
            NPP_BORDER_REPLICATE));
    }
    // declare a host image for the result
    npp::ImageCPU_8u_C3 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
    // save host image to result file
    pImageSetter->saveImage(sResultFilename, oHostDst.data(),
                    oHostDst.pitch(), oHostDst.height(), oHostDst.width());
    // std::cout << "Saved image: " << sResultFilename << std::endl;

    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());
}

void NppProcessImage::ProcessC4Image(npp::NppRetrieveImage *pImageSetter,
                              std::string sResultFilename, int nBitDepth) {
    npp::ImageCPU_8u_C4 oHostSrc;
    // load gray-scale image from disk
    pImageSetter->loadImage(&oHostSrc, nBitDepth);
    npp::ImageNPP_8u_C4 oDeviceSrc(oHostSrc);
    NppiSize oSrcSize = {static_cast<int>(oDeviceSrc.width()),
                        static_cast<int>(oDeviceSrc.height())};
    // create struct with ROI size
    NppiSize oSizeROI = {static_cast<int>(oDeviceSrc.width()),
                        static_cast<int>(oDeviceSrc.height())};
    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C4 oDeviceDst(oSizeROI.width, oSizeROI.height);
    if (nFilterType == FilterType_FilterBoxBorder) {
        // run box filter
        NPP_CHECK_NPP(nppiFilterBoxBorder_8u_C4R(
            oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
            oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, oMaskSize, oAnchor,
            NPP_BORDER_REPLICATE));
    } else if (nFilterType == FilterType_FilterGaussBorder) {
        Npp32s nSrcStep = oDeviceSrc.pitch();
        Npp32s nDstStep = oDeviceDst.pitch();

        // run gauss border filter
        NPP_CHECK_NPP(nppiFilterGaussBorder_8u_C4R(
            oDeviceSrc.data(), nSrcStep, oSrcSize, oSrcOffset,
            oDeviceDst.data(), nDstStep, oSizeROI, oGaussMaskSize,
            NPP_BORDER_REPLICATE));
    }
    // declare a host image for the result
    npp::ImageCPU_8u_C4 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
    // save host image to result file
    pImageSetter->saveImage(sResultFilename, oHostDst.data(),
                    oHostDst.pitch(), oHostDst.height(), oHostDst.width());
    // std::cout << "Saved image: " << sResultFilename << std::endl;

    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());
}

void NppProcessImage::SetMaskSize(int width, int height) {
    oMaskSize.width = width;
    oMaskSize.height = height;
}
void NppProcessImage::SetSrcOffset(int x, int y) {
    oSrcOffset.x = x;
    oSrcOffset.y = y;
}
void NppProcessImage::SetAnchor(int x, int y) {
    oAnchor.x = x;
    oAnchor.y = y;
}

void NppProcessImage::SetGaussMaskSize(int nMaskSize) {
    switch (nMaskSize) {
    case 0:
        oGaussMaskSize = NPP_MASK_SIZE_1_X_3;
        break;
    case 1:
        oGaussMaskSize = NPP_MASK_SIZE_1_X_5;
        break;
    case 2:
        oGaussMaskSize = NPP_MASK_SIZE_3_X_1;
        break;
    case 3:
        oGaussMaskSize = NPP_MASK_SIZE_5_X_1;
        break;
    case 4:
        oGaussMaskSize = NPP_MASK_SIZE_3_X_3;
        break;
    case 5:
        oGaussMaskSize = NPP_MASK_SIZE_5_X_5;
        break;
    case 6:
        oGaussMaskSize = NPP_MASK_SIZE_7_X_7;
        break;
    case 7:
        oGaussMaskSize = NPP_MASK_SIZE_9_X_9;
        break;
    case 8:
        oGaussMaskSize = NPP_MASK_SIZE_11_X_11;
        break;
    case 9:
        oGaussMaskSize = NPP_MASK_SIZE_13_X_13;
        break;
    case 10:
        oGaussMaskSize = NPP_MASK_SIZE_15_X_15;
        break;
    default:
        oGaussMaskSize = NPP_MASK_SIZE_5_X_5;
        break;
    }
}

void NppProcessImage::SetFilterType(enumImageFilterType nType) {
    nFilterType = nType;
}

void NppProcessImage::ProcessImageNPP(npp::NppRetrieveImage *pImageSetter,
                            std::string szResultFileName, int nBitDepth) {
    if (nBitDepth == 8) {
        ProcessC1Image(pImageSetter, szResultFileName, nBitDepth);
    } else if (nBitDepth == 16) {
        ProcessC2Image(pImageSetter, szResultFileName, nBitDepth);
    } else if (nBitDepth == 24) {
        ProcessC3Image(pImageSetter, szResultFileName, nBitDepth);
    } else if (nBitDepth == 32) {
        ProcessC4Image(pImageSetter, szResultFileName, nBitDepth);
    }
}
