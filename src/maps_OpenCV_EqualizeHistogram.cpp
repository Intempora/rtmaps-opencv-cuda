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
// Date: 2019
////////////////////////////////

////////////////////////////////
// Purpose of this module : Equalizes histogram of grayscale image.
////////////////////////////////


#include "maps_OpenCV_EqualizeHistogram.h"  // Includes the header of this component

#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudawarping.hpp>
#include "opencv2/cudaarithm.hpp"

// Use the macros to declare the inputs
MAPS_BEGIN_INPUTS_DEFINITION(MAPSOpenCV_EqualizeHistogram)
MAPS_INPUT("imageIn", MAPS::FilterIplImage, MAPS::FifoReader)
MAPS_INPUT("i_gpu", Filter_MapsCudaStruct, MAPS::FifoReader)
MAPS_END_INPUTS_DEFINITION

// Use the macros to declare the outputs
MAPS_BEGIN_OUTPUTS_DEFINITION(MAPSOpenCV_EqualizeHistogram)
MAPS_OUTPUT("imageOut", MAPS::IplImage, nullptr, nullptr, 0)
MAPS_OUTPUT_USER_DYNAMIC_STRUCTURE("o_gpu", MapsCudaStruct)
MAPS_END_OUTPUTS_DEFINITION

// Use the macros to declare the properties
MAPS_BEGIN_PROPERTIES_DEFINITION(MAPSOpenCV_EqualizeHistogram)
MAPS_PROPERTY("use_cuda", false, false, false)
MAPS_PROPERTY("gpu_mat_as_input", false, false, false)
MAPS_PROPERTY("gpu_mat_as_output", false, false, false)
MAPS_END_PROPERTIES_DEFINITION

// Use the macros to declare the actions
MAPS_BEGIN_ACTIONS_DEFINITION(MAPSOpenCV_EqualizeHistogram)
    //MAPS_ACTION("aName",MAPSOpenCV_EqualizeHistogram::ActionName)
MAPS_END_ACTIONS_DEFINITION

// Use the macros to declare this component (OpenCV_Resize) behaviour
MAPS_COMPONENT_DEFINITION(MAPSOpenCV_EqualizeHistogram,"OpenCV_HistogramEqualize_cuda", "1.1.1", 128,
                            MAPS::Threaded|MAPS::Sequential, MAPS::Threaded,
                            0, // Nb of inputs
                            0, // Nb of outputs
                            1, // Nb of properties
                            -1) // Nb of actions


void MAPSOpenCV_EqualizeHistogram::Birth()
{
    if (m_useCuda && m_gpuMatAsInput)
    {
        m_inputReader = MAPS::MakeInputReader::Reactive(
            this,
            Input(0),
            &MAPSOpenCV_EqualizeHistogram::AllocateOutputBufferSizeGpu,  // Called when data is received for the first time only
            &MAPSOpenCV_EqualizeHistogram::ProcessDataGpu      // Called when data is received for the first time AND all subsequent times
        );
    }
    else
    {
        m_inputReader = MAPS::MakeInputReader::Reactive(
            this,
            Input(0),
            &MAPSOpenCV_EqualizeHistogram::AllocateOutputBufferSize,  // Called when data is received for the first time only
            &MAPSOpenCV_EqualizeHistogram::ProcessData      // Called when data is received for the first time AND all subsequent times
        );
    }
}

void MAPSOpenCV_EqualizeHistogram::Core()
{
    m_inputReader->Read();
}

void MAPSOpenCV_EqualizeHistogram::Death()
{
    m_inputReader.reset();
}

void MAPSOpenCV_EqualizeHistogram::Dynamic()
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

void MAPSOpenCV_EqualizeHistogram::FreeBuffers()
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

void MAPSOpenCV_EqualizeHistogram::AllocateOutputBufferSize(const MAPSTimestamp, const MAPS::InputElt<IplImage> imageInElt)
{
    const IplImage& imageIn = imageInElt.Data();

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

void MAPSOpenCV_EqualizeHistogram::AllocateOutputBufferSizeGpu(const MAPSTimestamp, const MAPS::InputElt<MapsCudaStruct> imageInElt)
{
    const MapsCudaStruct& imageIn = imageInElt.Data();

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

void MAPSOpenCV_EqualizeHistogram::ProcessData(const MAPSTimestamp ts, const MAPS::InputElt<IplImage> inElt)
{
    try
    {
        const IplImage& imageIn = inElt.Data();
        m_tempImageIn = convTools::noCopyIplImage2Mat(&imageIn); // Convert IplImage to cv::Mat without copying
        MAPS::OutputGuard<> outGuard{ this, Output(0) };

        if (m_useCuda)
        {
            cv::cuda::GpuMat src(m_tempImageIn);
            if (m_gpuMatAsOutput)
            {
                MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
                const IplImage& proxyDst = outputData.m_IplImageProxy;
                cv::cuda::GpuMat dst(proxyDst.height, proxyDst.width, CV_MAKETYPE(proxyDst.depth == 8 ? 0 : proxyDst.depth / 8, proxyDst.nChannels), outputData.m_points);
                if (m_tempImageIn.channels() == 1) // If there is only one channel, equalize it
                {
                    cv::cuda::equalizeHist(src, dst);
                }
                else
                {
                    std::vector<cv::cuda::GpuMat> planesMatImages;
                    cv::cuda::split(src, planesMatImages); // Split the channels
                    for (int i = 0; i < imageIn.nChannels; i++)
                    {
                        cv::cuda::equalizeHist(planesMatImages[i], planesMatImages[i]); // Equalize all single-channel
                    }
                    cv::cuda::merge(planesMatImages, dst); // Merge to produce the equalize image
                }
            }
            else
            {
                cv::cuda::GpuMat dst;
                const IplImage& imageOut = outGuard.DataAs<IplImage>();
                m_tempImageOut = convTools::noCopyIplImage2Mat(&imageOut); // Convert IplImage to cv::Mat without copying

                if (m_tempImageIn.channels() == 1) // If there is only one channel, equalize it
                {
                    cv::cuda::equalizeHist(src, dst);
                }
                else
                {
                    std::vector<cv::cuda::GpuMat> planesMatImages;
                    cv::cuda::split(src, planesMatImages); // Split the channels
                    for (int i = 0; i < imageIn.nChannels; i++)
                    {
                        cv::cuda::equalizeHist(planesMatImages[i], planesMatImages[i]); // Equalize all single-channel
                    }
                    cv::cuda::merge(planesMatImages, dst); // Merge to produce the equalize image
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
            if (m_tempImageIn.channels() == 1) // If there is only one channel, equalize it
            {
                cv::equalizeHist(m_tempImageIn, m_tempImageOut);
            }
            else
            {
                cv::split(m_tempImageIn, m_planesMatImages); // Split the channels
                for (int i = 0; i < imageIn.nChannels; i++)
                {
                    cv::equalizeHist(m_planesMatImages[i], m_planesMatImages[i]); // Equalize all single-channel
                }
                cv::merge(m_planesMatImages, m_tempImageOut); // Merge to produce the equalize image
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

void MAPSOpenCV_EqualizeHistogram::ProcessDataGpu(const MAPSTimestamp ts, const MAPS::InputElt<MapsCudaStruct> inElt)
{
    try
    {
        MAPS::OutputGuard<> outGuard{ this, Output(0) };
        const IplImage& proxy = inElt.Data().m_IplImageProxy;
        const cv::cuda::GpuMat src(proxy.height, proxy.width, CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels), inElt.Data().m_points);
        const IplImage& imageIn = inElt.Data().m_IplImageProxy;

        if (m_gpuMatAsOutput)
        {
            MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
            const IplImage& proxyDst = outputData.m_IplImageProxy;
            cv::cuda::GpuMat dst(proxyDst.height, proxyDst.width, CV_MAKETYPE(proxyDst.depth == 8 ? 0 : proxyDst.depth / 8, proxyDst.nChannels), outputData.m_points);
            if (imageIn.nChannels == 1) // If there is only one channel, equalize it
            {
                cv::cuda::equalizeHist(src, dst);
            }
            else
            {
                std::vector<cv::cuda::GpuMat> planesMatImages;
                cv::cuda::split(src, planesMatImages); // Split the channels
                for (int i = 0; i < imageIn.nChannels; i++)
                {
                    cv::cuda::equalizeHist(planesMatImages[i], planesMatImages[i]); // Equalize all single-channel
                }
                cv::cuda::merge(planesMatImages, dst); // Merge to produce the equalize image
            }
        }
        else
        {
            cv::cuda::GpuMat dst;
            const IplImage& imageOut = outGuard.DataAs<IplImage>();
            m_tempImageOut = convTools::noCopyIplImage2Mat(&imageOut); // Convert IplImage to cv::Mat without copying

            if (imageIn.nChannels == 1) // If there is only one channel, equalize it
            {
                cv::cuda::equalizeHist(src, dst);
            }
            else
            {
                std::vector<cv::cuda::GpuMat> planesMatImages;
                cv::cuda::split(src, planesMatImages); // Split the channels
                for (int i = 0; i < imageIn.nChannels; i++)
                {
                    cv::cuda::equalizeHist(planesMatImages[i], planesMatImages[i]); // Equalize all single-channel
                }
                cv::cuda::merge(planesMatImages, dst); // Merge to produce the equalize image
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
