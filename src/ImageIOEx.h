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
 * This file here was adapted from the original Nvidia "ImageIO.h" file 
 * available from the Cuda Sample folder. The changes here are governed
 * by the MIT license:
 *    Copyright(c) 2023 John Sogade 
 */

#ifndef SRC_IMAGEIOEX_H_
#define SRC_IMAGEIOEX_H_

#include "ImagesCPU.h"
#include "ImagesNPP.h"

#include "FreeImage.h"
#include "Exceptions.h"

#include <string>
#include <tuple>
#include <memory>
#include "/usr/include/string.h"

namespace npp {

class NppRetrieveImage{
        // Error handler for FreeImage library.
        //  In case this handler is invoked, it throws an NPP exception.
        static void
        FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char *zMessage) {
            throw npp::Exception(zMessage);
        }

        FIBITMAP *m_pBitmap;
        std::unique_ptr<ImageCPU_8u_C1> p_oImageC1;
        std::unique_ptr<ImageCPU_8u_C2> p_oImageC2;
        std::unique_ptr<ImageCPU_8u_C3> p_oImageC3;
        std::unique_ptr<ImageCPU_8u_C4> p_oImageC4;
        FREE_IMAGE_FORMAT m_eFormat;
        std::string m_fileExt = "PGM";
        int m_bitDepth = 8;

 public:
        // This function sets up the image bitmap and retrieves other
        // properties such as bit depth and file extension
        std::tuple<int, std::string> ImageSetup(const std::string &rFileName) {
            m_eFormat = FreeImage_GetFileType(rFileName.c_str());

            // no signature? try to guess the file format from the file
            // extension
            if (m_eFormat == FIF_UNKNOWN) {
                m_eFormat = FreeImage_GetFIFFromFilename(rFileName.c_str());
            }

            m_fileExt = rFileName.substr(rFileName.find_last_of("."));

            NPP_ASSERT(m_eFormat != FIF_UNKNOWN);
            // check that the plugin has reading capabilities ...

            if (FreeImage_FIFSupportsReading(m_eFormat)) {
                m_pBitmap = FreeImage_Load(m_eFormat, rFileName.c_str());
            }

            NPP_ASSERT(m_pBitmap != 0);

            m_bitDepth = FreeImage_GetBPP(m_pBitmap);

            switch (m_bitDepth) {
            case 8:
                p_oImageC1 = std::unique_ptr<ImageCPU_8u_C1>(
                    new ImageCPU_8u_C1(FreeImage_GetWidth(m_pBitmap),
                    FreeImage_GetHeight(m_pBitmap)));
                break;
            case 16:
                p_oImageC2 = std::unique_ptr<ImageCPU_8u_C2>(
                    new ImageCPU_8u_C2(FreeImage_GetWidth(m_pBitmap),
                    FreeImage_GetHeight(m_pBitmap)));
                break;
            case 24:
                p_oImageC3 = std::unique_ptr<ImageCPU_8u_C3>(
                    new ImageCPU_8u_C3(FreeImage_GetWidth(m_pBitmap),
                    FreeImage_GetHeight(m_pBitmap)));
                break;
            case 32:
                p_oImageC4 = std::unique_ptr<ImageCPU_8u_C4>(new ImageCPU_8u_C4(
                    FreeImage_GetWidth(m_pBitmap),
                    FreeImage_GetHeight(m_pBitmap)));
                break;

            default:
                break;
            }

            return {m_bitDepth, m_fileExt};
        }

        // Load a * channel gray-scale/color image from disk.
        void
        loadImage(void *rImage, int nbitDepth) {
            NPP_ASSERT_MSG(rImage != NULL, "ImageCPU_8u_C* pointer is NULL");
            // set your own FreeImage error handler
            FreeImage_SetOutputMessage(FreeImageErrorHandler);

            loadImage(nbitDepth);

            // swap the user given image with our result image, effecively
            // moving our newly loaded image data into the user provided shell
            if (nbitDepth == 8) {
                p_oImageC1->swap(*(reinterpret_cast<ImageCPU_8u_C1 *>(rImage)));
            } else if (nbitDepth == 24) {
                p_oImageC3->swap(*(reinterpret_cast<ImageCPU_8u_C3 *>(rImage)));
            } else if (nbitDepth == 32) {
                p_oImageC4->swap(*(reinterpret_cast<ImageCPU_8u_C4 *>(rImage)));
            }
        }

        // Load a * channel gray-scale/color image from disk.
        void
        loadImageNPP(void *rImage, int nbitDepth) {
            NPP_ASSERT_MSG(rImage != NULL, "ImageNPP_8u_C* pointer is NULL");
            switch (nbitDepth) {
            case 8:
            {
                ImageCPU_8u_C1 oImage;
                loadImage(&oImage, nbitDepth);
                ImageNPP_8u_C1 oResult(oImage);
                (reinterpret_cast<ImageNPP_8u_C1 *>(rImage))->swap(oResult);
            }
                break;
            case 16:
            {
                ImageCPU_8u_C2 oImage;
                loadImage(&oImage, nbitDepth);
                ImageNPP_8u_C2 oResult(oImage);
                (reinterpret_cast<ImageNPP_8u_C2 *>(rImage))->swap(oResult);
            }
                break;
            case 24:
            {
                ImageCPU_8u_C3 oImage;
                loadImage(&oImage, nbitDepth);
                ImageNPP_8u_C3 oResult(oImage);
                (reinterpret_cast<ImageNPP_8u_C3 *>(rImage))->swap(oResult);
            }
                break;
            case 32:
            {
                ImageCPU_8u_C4 oImage;
                loadImage(&oImage, nbitDepth);
                ImageNPP_8u_C4 oResult(oImage);
                (reinterpret_cast<ImageNPP_8u_C4 *>(rImage))->swap(oResult);
            }
                break;

            default:
                break;
            }
        }


    //******************************************************************************//
    // Load a * channel gray-scale/color image from disk.
    void
    loadImage(int nbitDepth) {
    const int nBytesPerPixel = nbitDepth / 8;
    // set your own FreeImage error handler
    FreeImage_SetOutputMessage(FreeImageErrorHandler);

    // make sure this is an 8-bit single channel image
    NPP_ASSERT(FreeImage_GetColorType(
                   m_pBitmap) == (nbitDepth == 8 ? FIC_MINISBLACK : FIC_RGB));

    // Copy the FreeImage data into the new ImageCPU
    // std::cout << "Did FreeImage_Load " << std::endl;
    unsigned int nSrcPitch = FreeImage_GetPitch(m_pBitmap);
    const Npp8u *pSrcLine = reinterpret_cast<Npp8u *>(
                                FreeImage_GetBits(m_pBitmap)) +
                            nSrcPitch * (FreeImage_GetHeight(m_pBitmap) - 1);

    Npp8u *pDstLine = NULL;
    unsigned int nDstPitch = -1;
    int nHeight = 0, nWidth = 0;

    // use the already created ImageCPU to receive the loaded image data
    switch (nbitDepth) {
    case 8:
        pDstLine = reinterpret_cast<Npp8u *>(p_oImageC1->data());
        nDstPitch = p_oImageC1->pitch();
        nHeight = p_oImageC1->height();
        nWidth = p_oImageC1->width();
        break;
    case 16:
        pDstLine = reinterpret_cast<Npp8u *>(p_oImageC2->data());
        nDstPitch = p_oImageC2->pitch();
        nHeight = p_oImageC2->height();
        nWidth = p_oImageC2->width();
        break;
    case 24:
        pDstLine = reinterpret_cast<Npp8u *>(p_oImageC3->data());
        nDstPitch = p_oImageC3->pitch();
        nHeight = p_oImageC3->height();
        nWidth = p_oImageC3->width();
        break;
    case 32:
        pDstLine = reinterpret_cast<Npp8u *>(p_oImageC4->data());
        nDstPitch = p_oImageC4->pitch();
        nHeight = p_oImageC4->height();
        nWidth = p_oImageC4->width();
        break;

    default:
        break;
    }

    for (size_t iLine = 0; iLine < nHeight; ++iLine) {
        memcpy(pDstLine, pSrcLine, nWidth * sizeof(Npp8u) * nBytesPerPixel);
        pSrcLine -= nSrcPitch;
        pDstLine += nDstPitch;
    }
}

        //******************************************************************************//
        //******************************************************************************//

        // Save a gray-scale image to disk.
        void
        saveImage(const std::string &rFileName, const ImageNPP_8u_C1 &rImage) {
            ImageCPU_8u_C1 oHostImage(rImage.size());
            // copy the device result data
            rImage.copyTo(oHostImage.data(), oHostImage.pitch());
            saveImage(
                rFileName, oHostImage.data(), oHostImage.pitch(),
                oHostImage.height(), oHostImage.width());
        }

        // Save an 3 channel color image to disk.
        void
        saveImage(const std::string &rFileName, const ImageNPP_8u_C3 &rImage) {
            ImageCPU_8u_C3 oHostImage(rImage.size());
            // copy the device result data
            rImage.copyTo(oHostImage.data(), oHostImage.pitch());
            saveImage(
                rFileName, oHostImage.data(), oHostImage.pitch(),
                oHostImage.height(), oHostImage.width());
        }

        // Save a 4 channel color image to disk.
        void
        saveImage(const std::string &rFileName, const ImageNPP_8u_C4 &rImage) {
            ImageCPU_8u_C4 oHostImage(rImage.size());
            // copy the device result data
            rImage.copyTo(oHostImage.data(), oHostImage.pitch());
            saveImage(
                rFileName, oHostImage.data(), oHostImage.pitch(),
                oHostImage.height(), oHostImage.width());
        }

        // Save a * channel gray-scale/color image to disk.
        void
        saveImage(
            const std::string &rFileName, Npp8u *pSrcLine,
            unsigned int nSrcPitch, int nHeight, int nWidth) {
            const int nBytesPerPixel = m_bitDepth / 8;

            // create the result image storage using FreeImage
            // so we can easily save
            FIBITMAP *pResultBitmap = FreeImage_Allocate(
                        nWidth, nHeight, m_bitDepth /* bits per pixel */);
            NPP_ASSERT_NOT_NULL(pResultBitmap);
            unsigned int nDstPitch = FreeImage_GetPitch(pResultBitmap);
            Npp8u *pDstLine =
                reinterpret_cast<Npp8u *>(FreeImage_GetBits(pResultBitmap)) +
                nDstPitch * (nHeight - 1);

            for (size_t iLine = 0; iLine < nHeight; ++iLine) {
                memcpy(pDstLine, pSrcLine, nWidth * sizeof(Npp8u) *
                    nBytesPerPixel);
                pSrcLine += nSrcPitch;
                pDstLine -= nDstPitch;
            }

            // now save the result image
            bool bSuccess;
            bSuccess =
            FreeImage_Save(m_eFormat, pResultBitmap, rFileName.c_str(), 0) ==
                TRUE;
            NPP_ASSERT_MSG(bSuccess, "Failed to save result image.");
        }
};
}  // namespace npp
#endif  //  SRC_IMAGEIOEX_H_
