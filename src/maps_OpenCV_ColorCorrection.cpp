/////////////////////////////////////////////////////////////////////////////////
//
//   Copyright 2014-2024 Intempora S.A.S.
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
/////////////////////////////////////////////////////////////////////////////////

////////////////////////////////
// Author: Intempora S.A. - NL
// Date: 2023
////////////////////////////////

////////////////////////////////
// Purpose of this module : This component applies a gain on each channel separately to correct colors.
////////////////////////////////

#include "maps_OpenCV_ColorCorrection.h"	// Includes the header of this component

#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudawarping.hpp>
#include "opencv2/cudaarithm.hpp"

// Use the macros to declare the inputs
MAPS_BEGIN_INPUTS_DEFINITION(MAPSColorCorrection)
MAPS_INPUT("imageIn", MAPS::FilterIplImage, MAPS::FifoReader)
MAPS_INPUT("i_gpu", Filter_MapsCudaStruct, MAPS::FifoReader)
MAPS_END_INPUTS_DEFINITION

// Use the macros to declare the outputs
MAPS_BEGIN_OUTPUTS_DEFINITION(MAPSColorCorrection)
MAPS_OUTPUT("imageOut", MAPS::IplImage, nullptr, nullptr, 0)
MAPS_OUTPUT_USER_DYNAMIC_STRUCTURE("o_gpu", MapsCudaStruct)
MAPS_END_OUTPUTS_DEFINITION

// Use the macros to declare the properties
MAPS_BEGIN_PROPERTIES_DEFINITION(MAPSColorCorrection)
    MAPS_PROPERTY("red", 1.0, false, true)
    MAPS_PROPERTY("green", 1.0, false, true)
    MAPS_PROPERTY("blue", 1.0, false, true)
    MAPS_PROPERTY("use_cuda", false, false, false)
    MAPS_PROPERTY("gpu_mat_as_input", false, false, false)
    MAPS_PROPERTY("gpu_mat_as_output", false, false, false)
MAPS_END_PROPERTIES_DEFINITION

// Use the macros to declare the actions
MAPS_BEGIN_ACTIONS_DEFINITION(MAPSColorCorrection)
MAPS_END_ACTIONS_DEFINITION

// Use the macros to declare this component behaviour
MAPS_COMPONENT_DEFINITION(MAPSColorCorrection,"OpenCV_ColorCorrection_cuda", "1.2.1", 128,
                            MAPS::Threaded|MAPS::Sequential, MAPS::Sequential,
                            0, // Nb of inputs
                            0, // Nb of outputs
                            4, // Nb of properties
                            -1) // Nb of actions


void MAPSColorCorrection::Birth()
{
    if (m_useCuda && m_gpuMatAsInput)
    {
        m_inputReader = MAPS::MakeInputReader::Reactive(
            this,
            Input(0),
            &MAPSColorCorrection::AllocateOutputBufferSizeGpu,  // Called when data is received for the first time only
            &MAPSColorCorrection::ProcessDataGpu      // Called when data is received for the first time AND all subsequent times
        );
    }
    else
    {
        m_inputReader = MAPS::MakeInputReader::Reactive(
            this,
            Input(0),
            &MAPSColorCorrection::AllocateOutputBufferSize,  // Called when data is received for the first time only
            &MAPSColorCorrection::ProcessData      // Called when data is received for the first time AND all subsequent times
        );
    }

    m_dRed = GetFloatProperty("red");
    m_dGreen = GetFloatProperty("green");
    m_dBlue = GetFloatProperty("blue");
}

void MAPSColorCorrection::Core()
{
    m_inputReader->Read();
}

void MAPSColorCorrection::Death()
{
    m_inputReader.reset();
}

void MAPSColorCorrection::Dynamic()
{
    m_useCuda = false;
    m_gpuMatAsInput = false;
    m_gpuMatAsOutput = false;

    if (cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        Property("use_cuda").SetMutable(true);
    }
    else
    {
        Property("use_cuda").SetMutable(false);
    }

    if (Property("use_cuda").IsMutable())
        m_useCuda = GetBoolProperty("use_cuda");

    if (m_useCuda)
    {
        m_gpuMatAsInput = NewProperty("gpu_mat_as_input").BoolValue();
        m_gpuMatAsOutput = NewProperty("gpu_mat_as_output").BoolValue();

        if (m_gpuMatAsInput)
        {
            NewInput("i_gpu");
        }
        else
        {
            NewInput("imageIn");
        }

        if (m_gpuMatAsOutput)
        {
            NewOutput("o_gpu");
        }
        else
        {
            NewOutput("imageOut");
        }
    }
    else
    {
        NewInput("imageIn");
        NewOutput("imageOut");
    }
}

void MAPSColorCorrection::FreeBuffers()
{
    if (m_useCuda && m_gpuMatAsOutput)
    {
        MAPS_DynamicCustomStructComponent::FreeBuffers();
    }
    else
    {
        MAPSComponent::FreeBuffers();
    }
}

void MAPSColorCorrection::Set(MAPSProperty &p, MAPSFloat64 value)
{
    MAPSComponent::Set(p, value);
    if (p.ShortName() == "red")
        m_dRed = value;
    else if (p.ShortName() == "green")
        m_dGreen = value;
    else if (p.ShortName() == "blue")
        m_dBlue = value;
}

void MAPSColorCorrection::AllocateOutputBufferSize(const MAPSTimestamp, const MAPS::InputElt<IplImage> imageInElt)
{
    const IplImage& imageIn = imageInElt.Data();
    const MAPSInt32 chanSeq = *(MAPSInt32*)imageIn.channelSeq;

    if (chanSeq != MAPS_CHANNELSEQ_BGR && chanSeq != MAPS_CHANNELSEQ_BGRA && chanSeq != MAPS_CHANNELSEQ_RGB &&
        chanSeq != MAPS_CHANNELSEQ_RGBA)
        Error("This component only accepts RGB/BGR/RGBA/BGRA images on its input.");

    if (m_gpuMatAsOutput)
    {
        try
        {
            AllocateDynamicOutputBuffers(
                DynamicOutput<MapsCudaStruct>(Output("o_gpu"),
                    [&] { return new MapsCudaStruct(imageIn.width, imageIn.height, imageIn.nChannels, imageIn); }  // struct allocation
                )
            );
        }
        catch (...)
        {
            Error("Failed to allocate the dynamic output buffers");
        }
    }
    else
    {
        Output(0).AllocOutputBufferIplImage(imageIn);
    }
}

void MAPSColorCorrection::AllocateOutputBufferSizeGpu(const MAPSTimestamp, const MAPS::InputElt<MapsCudaStruct> imageInElt)
{
    const MapsCudaStruct& imageIn = imageInElt.Data();

    const MAPSInt32 chanSeq = *(MAPSInt32*)imageIn.m_IplImageProxy.channelSeq;

    if (chanSeq != MAPS_CHANNELSEQ_BGR && chanSeq != MAPS_CHANNELSEQ_BGRA && chanSeq != MAPS_CHANNELSEQ_RGB &&
        chanSeq != MAPS_CHANNELSEQ_RGBA)
        Error("This component only accepts RGB/BGR/RGBA/BGRA images on its input.");

    if (m_gpuMatAsOutput)
    {
        try
        {
            AllocateDynamicOutputBuffers(
                DynamicOutput<MapsCudaStruct>(Output("o_gpu"),
                    [&] { return new MapsCudaStruct(imageIn); }
                )
            );
        }
        catch (...)
        {
            Error("Failed to allocate the dynamic output buffers");
        }
    }
    else
    {
        IplImage model = MAPS::IplImageModel(imageIn.m_IplImageProxy.width, imageIn.m_IplImageProxy.height, imageIn.m_IplImageProxy.channelSeq, IPL_DATA_ORDER_PIXEL,
            imageIn.m_IplImageProxy.depth, imageIn.m_IplImageProxy.align);
        Output(0).AllocOutputBufferIplImage(model);
    }
}

void MAPSColorCorrection::ProcessData(const MAPSTimestamp ts, const MAPS::InputElt<IplImage> inElt)
{
    try
    {
        const IplImage& imageIn = inElt.Data();
        m_tempImageIn = convTools::noCopyIplImage2Mat(&imageIn); // Convert IplImage to cv::Mat without copying
        MAPS::OutputGuard<> outGuard{ this, Output(0) };
        const MAPSInt32 chanSeq = *(MAPSInt32*)imageIn.channelSeq;

        if (m_useCuda)
        {
            cv::cuda::GpuMat src(m_tempImageIn);
            std::vector<cv::cuda::GpuMat> vTmpSplit;

            if (m_gpuMatAsOutput)
            {
                MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
                const IplImage& proxy = outputData.m_IplImageProxy;
                cv::cuda::GpuMat dst(proxy.height, proxy.width, CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels), outputData.m_points);

                if (chanSeq == MAPS_CHANNELSEQ_BGR || chanSeq == MAPS_CHANNELSEQ_BGRA)
                {
                    cv::Scalar coefficients(m_dBlue, m_dGreen, m_dRed);
                    cv::cuda::multiply(src, coefficients, dst);
                }
                else
                {
                    cv::Scalar coefficients(m_dRed, m_dGreen, m_dBlue);
                    cv::cuda::multiply(src, coefficients, dst);
                }
            }
            else
            {
                cv::cuda::GpuMat dst;
                const IplImage& imageOut = outGuard.DataAs<IplImage>();
                m_tempImageOut = convTools::noCopyIplImage2Mat(&imageOut); // Convert IplImage to cv::Mat without copying

                if (chanSeq == MAPS_CHANNELSEQ_BGR || chanSeq == MAPS_CHANNELSEQ_BGRA)
                {
                    cv::Scalar coefficients(m_dBlue, m_dGreen, m_dRed);
                    cv::cuda::multiply(src, coefficients, dst);
                }
                else
                {
                    cv::Scalar coefficients(m_dRed, m_dGreen, m_dBlue);
                    cv::cuda::multiply(src, coefficients, dst);
                }
                dst.download(m_tempImageOut);

                if (static_cast<void*>(m_tempImageOut.data) != static_cast<void*>(imageOut.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
                    Error("cv::Mat data ptr and imageOut data ptr are different.");
            }
        }
        else
        {
            const IplImage& imageOut = outGuard.DataAs<IplImage>();
            m_tempImageOut = convTools::noCopyIplImage2Mat(&imageOut); // Convert IplImage to cv::Mat without copying

            if (chanSeq == MAPS_CHANNELSEQ_BGR || chanSeq == MAPS_CHANNELSEQ_BGRA)
            {
                cv::Scalar coefficients(m_dBlue, m_dGreen, m_dRed);
                cv::multiply(m_tempImageIn, coefficients, m_tempImageOut);
            }
            else
            {
                cv::Scalar coefficients(m_dRed, m_dGreen, m_dBlue);
                cv::multiply(m_tempImageIn, coefficients, m_tempImageOut);
            }

            if (static_cast<void*>(m_tempImageOut.data) != static_cast<void*>(imageOut.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
                Error("cv::Mat data ptr and imageOut data ptr are different.");
        }

        outGuard.Timestamp() = ts;
    }
    catch (const std::exception& e)
    {
        Error(e.what());
    }
}

void MAPSColorCorrection::ProcessDataGpu(const MAPSTimestamp ts, const MAPS::InputElt<MapsCudaStruct> inElt)
{
    try
    {
        MAPS::OutputGuard<> outGuard{ this, Output(0) };

        const IplImage& proxy = inElt.Data().m_IplImageProxy;
        const cv::cuda::GpuMat src(proxy.height, proxy.width, CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels), inElt.Data().m_points);

        const MAPSInt32 chanSeq = *(MAPSInt32*)proxy.channelSeq;
        std::vector<cv::cuda::GpuMat> vTmpSplit;

        if (m_gpuMatAsOutput)
        {
            MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
            const IplImage& proxyDst = outputData.m_IplImageProxy;
            cv::cuda::GpuMat dst(proxyDst.height, proxyDst.width, CV_MAKETYPE(proxyDst.depth == 8 ? 0 : proxyDst.depth / 8, proxyDst.nChannels), outputData.m_points);

            if (chanSeq == MAPS_CHANNELSEQ_BGR || chanSeq == MAPS_CHANNELSEQ_BGRA)
            {
                cv::Scalar coefficients(m_dBlue, m_dGreen, m_dRed);
                cv::cuda::multiply(src, coefficients, dst);
            }
            else
            {
                cv::Scalar coefficients(m_dRed, m_dGreen, m_dBlue);
                cv::cuda::multiply(src, coefficients, dst);
            }
        }
        else
        {
            cv::cuda::GpuMat dst;
            const IplImage& imageOut = outGuard.DataAs<IplImage>();
            m_tempImageOut = convTools::noCopyIplImage2Mat(&imageOut); // Convert IplImage to cv::Mat without copying

            if (chanSeq == MAPS_CHANNELSEQ_BGR || chanSeq == MAPS_CHANNELSEQ_BGRA)
            {
                cv::Scalar coefficients(m_dBlue, m_dGreen, m_dRed);
                cv::cuda::multiply(src, coefficients, dst);
            }
            else
            {
                cv::Scalar coefficients(m_dRed, m_dGreen, m_dBlue);
                cv::cuda::multiply(src, coefficients, dst);
            }

            dst.download(m_tempImageOut);

            if (static_cast<void*>(m_tempImageOut.data) != static_cast<void*>(imageOut.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
                Error("cv::Mat data ptr and imageOut data ptr are different.");
        }
        
        outGuard.Timestamp() = ts;
    }
    catch (const std::exception& e)
    {
        Error(e.what());
    }
}
