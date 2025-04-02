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
// Purpose of this module : Divides 3-channel image into several single channel GRAY images.
////////////////////////////////

#include "maps_OpenCV_ChannelsSplitter.h"	// Includes the header of this component

#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudawarping.hpp>
#include "opencv2/cudaarithm.hpp"

// Use the macros to declare the inputs
MAPS_BEGIN_INPUTS_DEFINITION(MAPSOpenCV_SplitChannels)
    MAPS_INPUT("imageIn", MAPS::FilterIplImage, MAPS::FifoReader)
    MAPS_INPUT("i_gpu", Filter_MapsCudaStruct, MAPS::FifoReader)
MAPS_END_INPUTS_DEFINITION

// Use the macros to declare the outputs
MAPS_BEGIN_OUTPUTS_DEFINITION(MAPSOpenCV_SplitChannels)
    MAPS_OUTPUT("channel1", MAPS::IplImage, nullptr, nullptr, 0)
    MAPS_OUTPUT("channel2", MAPS::IplImage, nullptr, nullptr, 0)
    MAPS_OUTPUT("channel3", MAPS::IplImage, nullptr, nullptr, 0)
    MAPS_OUTPUT_USER_DYNAMIC_STRUCTURE("o_gpu_channel1", MapsCudaStruct)
    MAPS_OUTPUT_USER_DYNAMIC_STRUCTURE("o_gpu_channel2", MapsCudaStruct)
    MAPS_OUTPUT_USER_DYNAMIC_STRUCTURE("o_gpu_channel3", MapsCudaStruct)
MAPS_END_OUTPUTS_DEFINITION

// Use the macros to declare the properties
MAPS_BEGIN_PROPERTIES_DEFINITION(MAPSOpenCV_SplitChannels)
MAPS_PROPERTY("use_cuda", false, false, false)
MAPS_PROPERTY("gpu_mat_as_input", false, false, false)
MAPS_PROPERTY("gpu_mat_as_output", false, false, false)
MAPS_END_PROPERTIES_DEFINITION

// Use the macros to declare the actions
MAPS_BEGIN_ACTIONS_DEFINITION(MAPSOpenCV_SplitChannels)
    //MAPS_ACTION("aName",MAPSOpenCV_SplitChannels::ActionName)
MAPS_END_ACTIONS_DEFINITION

// Use the macros to declare this component (OpenCV_Resize) behaviour
MAPS_COMPONENT_DEFINITION(MAPSOpenCV_SplitChannels, "OpenCV_ChannelsSplitter_cuda", "1.1.0", 128,
                            MAPS::Threaded|MAPS::Sequential, MAPS::Threaded,
                            0, // Nb of inputs
                            0, // Nb of outputs
                            1, // Nb of properties
                            -1) // Nb of actions

void MAPSOpenCV_SplitChannels::Birth()
{
    if (m_useCuda && m_gpuMatAsInput)
    {
        m_inputReader = MAPS::MakeInputReader::Reactive(
            this,
            Input(0),
            &MAPSOpenCV_SplitChannels::AllocateOutputBufferSizeGpu,  // Called when data is received for the first time only
            &MAPSOpenCV_SplitChannels::ProcessDataGpu      // Called when data is received for the first time AND all subsequent times
        );
    }
    else
    {
        m_inputReader = MAPS::MakeInputReader::Reactive(
            this,
            Input(0),
            &MAPSOpenCV_SplitChannels::AllocateOutputBufferSize,  // Called when data is received for the first time only
            &MAPSOpenCV_SplitChannels::ProcessData      // Called when data is received for the first time AND all subsequent times
        );
    }
}

void MAPSOpenCV_SplitChannels::Core()
{
    m_inputReader->Read();
}

void MAPSOpenCV_SplitChannels::Death()
{
    m_inputReader.reset();
}

void MAPSOpenCV_SplitChannels::Dynamic()
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
            NewOutput("o_gpu_channel1");
            NewOutput("o_gpu_channel2");
            NewOutput("o_gpu_channel3");
        }
        else
        {
            NewOutput("channel1");
            NewOutput("channel2");
            NewOutput("channel3");
        }
    }
    else
    {
        NewInput("imageIn");
        NewOutput("channel1");
        NewOutput("channel2");
        NewOutput("channel3");
    }
}

void MAPSOpenCV_SplitChannels::FreeBuffers()
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

void MAPSOpenCV_SplitChannels::AllocateOutputBufferSize(const MAPSTimestamp, const MAPS::InputElt<IplImage> imageInElt)
{
    const IplImage& imageIn = imageInElt.Data();

    if (imageIn.nChannels != 3)
    {
        Error("This component only supports 3 channels images.");
    }

    m_isInputPlanar = (imageIn.dataOrder == IPL_DATA_ORDER_PLANE);
    IplImage model = MAPS::IplImageModel(imageIn.width, imageIn.height, MAPS_CHANNELSEQ_GRAY, imageIn.dataOrder, imageIn.depth, imageIn.align);

    if (m_gpuMatAsOutput)
    {
        try
        {
            AllocateDynamicOutputBuffers(
                DynamicOutput<MapsCudaStruct>(Output("o_gpu_channel1"),
                    [&] { return new MapsCudaStruct(imageIn.width, imageIn.height, 1, model); }  // struct allocation
                ),
                DynamicOutput<MapsCudaStruct>(Output("o_gpu_channel2"),
                    [&] { return new MapsCudaStruct(imageIn.width, imageIn.height, 1, model); }  // struct allocation
                ),
                DynamicOutput<MapsCudaStruct>(Output("o_gpu_channel3"),
                    [&] { return new MapsCudaStruct(imageIn.width, imageIn.height, 1, model); }  // struct allocation
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
        Output("channel1").AllocOutputBufferIplImage(model);
        Output("channel2").AllocOutputBufferIplImage(model);
        Output("channel3").AllocOutputBufferIplImage(model);
    }
}

void MAPSOpenCV_SplitChannels::ProcessData(const MAPSTimestamp ts, const MAPS::InputElt<IplImage> inElt)
{
    try
    {
        const IplImage& imageIn = inElt.Data();
        const cv::Mat tempImageIn = convTools::noCopyIplImage2Mat(&imageIn); // Convert IplImage to cv::Mat without copying
        MAPS::OutputGuard<> outGuard1{ this, Output(0) };
        MAPS::OutputGuard<> outGuard2{ this, Output(1) };
        MAPS::OutputGuard<> outGuard3{ this, Output(2) };

        if (m_useCuda)
        {
            cv::cuda::GpuMat src(tempImageIn);
            if (m_gpuMatAsOutput)
            {
                std::vector<cv::cuda::GpuMat> mats;
                cv::cuda::split(src, mats);

                MapsCudaStruct& outputData1 = outGuard1.DataAs<MapsCudaStruct>();
                MapsCudaStruct& outputData2 = outGuard2.DataAs<MapsCudaStruct>();
                MapsCudaStruct& outputData3 = outGuard3.DataAs<MapsCudaStruct>();
                const IplImage& proxy = outputData1.m_IplImageProxy;
                cv::cuda::GpuMat dst1(proxy.height, proxy.width, CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels), outputData1.m_points);
                cv::cuda::GpuMat dst2(proxy.height, proxy.width, CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels), outputData2.m_points);
                cv::cuda::GpuMat dst3(proxy.height, proxy.width, CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels), outputData3.m_points);

                mats[0].copyTo(dst1);
                mats[1].copyTo(dst2);
                mats[2].copyTo(dst3);
            }
            else
            {
                IplImage& imageOut1 = outGuard1.DataAs<IplImage>();
                IplImage& imageOut2 = outGuard2.DataAs<IplImage>();
                IplImage& imageOut3 = outGuard3.DataAs<IplImage>();
                m_tempImageOut[0] = convTools::noCopyIplImage2Mat(&imageOut1);
                m_tempImageOut[1] = convTools::noCopyIplImage2Mat(&imageOut2);
                m_tempImageOut[2] = convTools::noCopyIplImage2Mat(&imageOut3);

                std::vector<cv::cuda::GpuMat> mats;
                cv::cuda::split(src, mats);
                mats[0].download(m_tempImageOut[0]);
                mats[1].download(m_tempImageOut[1]);
                mats[2].download(m_tempImageOut[2]);

                if (static_cast<void*>(m_tempImageOut[0].data) != static_cast<void*>(imageOut1.imageData) ||
                    static_cast<void*>(m_tempImageOut[1].data) != static_cast<void*>(imageOut2.imageData) ||
                    static_cast<void*>(m_tempImageOut[2].data) != static_cast<void*>(imageOut3.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
                    Error("cv::Mat data ptr and imageOut data ptr are different.");
            }
        }
        else
        {
            IplImage& imageOut1 = outGuard1.DataAs<IplImage>();
            IplImage& imageOut2 = outGuard2.DataAs<IplImage>();
            IplImage& imageOut3 = outGuard3.DataAs<IplImage>();

            if (m_isInputPlanar)
            {
                std::memcpy(imageOut1.imageData, imageIn.imageData, imageOut1.imageSize);
                std::memcpy(imageOut2.imageData, imageIn.imageData + imageOut2.imageSize, imageOut2.imageSize);
                std::memcpy(imageOut3.imageData, imageIn.imageData + 2 * imageOut3.imageSize, imageOut3.imageSize);
            }
            else
            {
                m_tempImageOut[0] = convTools::noCopyIplImage2Mat(&imageOut1);
                m_tempImageOut[1] = convTools::noCopyIplImage2Mat(&imageOut2);
                m_tempImageOut[2] = convTools::noCopyIplImage2Mat(&imageOut3);
                cv::split(tempImageIn, m_tempImageOut);

                if (static_cast<void*>(m_tempImageOut[0].data) != static_cast<void*>(imageOut1.imageData) ||
                    static_cast<void*>(m_tempImageOut[1].data) != static_cast<void*>(imageOut2.imageData) ||
                    static_cast<void*>(m_tempImageOut[2].data) != static_cast<void*>(imageOut3.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
                    Error("cv::Mat data ptr and imageOut data ptr are different.");
            }
        }

        outGuard1.Timestamp() = ts;
        outGuard2.Timestamp() = ts;
        outGuard3.Timestamp() = ts;
    }
    catch (const std::exception& e)
    {
        Error(e.what());
    }
}

void MAPSOpenCV_SplitChannels::AllocateOutputBufferSizeGpu(const MAPSTimestamp, const MAPS::InputElt<MapsCudaStruct> imageInElt)
{
    const MapsCudaStruct& imageIn = imageInElt.Data();

    if (imageIn.m_IplImageProxy.nChannels != 3)
    {
        Error("This component only supports 3 channels images.");
    }

    const IplImage& proxy = imageIn.m_IplImageProxy;
    m_isInputPlanar = (proxy.dataOrder == IPL_DATA_ORDER_PLANE);
    IplImage model = MAPS::IplImageModel(proxy.width, proxy.height, MAPS_CHANNELSEQ_GRAY, proxy.dataOrder, proxy.depth, proxy.align);

    int type = CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels);

    if (m_gpuMatAsOutput)
    {
        try
        {
            AllocateDynamicOutputBuffers(
                DynamicOutput<MapsCudaStruct>(Output("o_gpu_channel1"),
                    [&] { return new MapsCudaStruct(proxy.width, proxy.height, 1, model); }
                ),
                DynamicOutput<MapsCudaStruct>(Output("o_gpu_channel2"),
                    [&] { return new MapsCudaStruct(proxy.width, proxy.height, 1, model); }
                ),
                DynamicOutput<MapsCudaStruct>(Output("o_gpu_channel3"),
                    [&] { return new MapsCudaStruct(proxy.width, proxy.height, 1, model); }
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
        Output("channel1").AllocOutputBufferIplImage(model);
        Output("channel2").AllocOutputBufferIplImage(model);
        Output("channel3").AllocOutputBufferIplImage(model);
    }
}

void MAPSOpenCV_SplitChannels::ProcessDataGpu(const MAPSTimestamp ts, const MAPS::InputElt<MapsCudaStruct> inElt)
{
    try
    {
        MAPS::OutputGuard<> outGuard1{ this, Output(0) };
        MAPS::OutputGuard<> outGuard2{ this, Output(1) };
        MAPS::OutputGuard<> outGuard3{ this, Output(2) };

        const IplImage& proxy = inElt.Data().m_IplImageProxy;
        const cv::cuda::GpuMat src(proxy.height, proxy.width, CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels), inElt.Data().m_points);

        if (m_gpuMatAsOutput)
        {
            std::vector<cv::cuda::GpuMat> mats;
            cv::cuda::split(src, mats);

            MapsCudaStruct& outputData = outGuard1.DataAs<MapsCudaStruct>();
            const IplImage& proxyDst = outputData.m_IplImageProxy;
            cv::cuda::GpuMat dst1(proxyDst.height, proxyDst.width, CV_MAKETYPE(proxyDst.depth == 8 ? 0 : proxyDst.depth / 8, proxyDst.nChannels), outputData.m_points);
            cv::cuda::GpuMat dst2(proxyDst.height, proxyDst.width, CV_MAKETYPE(proxyDst.depth == 8 ? 0 : proxyDst.depth / 8, proxyDst.nChannels), outputData.m_points);
            cv::cuda::GpuMat dst3(proxyDst.height, proxyDst.width, CV_MAKETYPE(proxyDst.depth == 8 ? 0 : proxyDst.depth / 8, proxyDst.nChannels), outputData.m_points);

            mats[0].copyTo(dst1);
            mats[1].copyTo(dst2);
            mats[2].copyTo(dst2);
        }
        else
        {
            IplImage& imageOut1 = outGuard1.DataAs<IplImage>();
            IplImage& imageOut2 = outGuard2.DataAs<IplImage>();
            IplImage& imageOut3 = outGuard3.DataAs<IplImage>();
            m_tempImageOut[0] = convTools::noCopyIplImage2Mat(&imageOut1);
            m_tempImageOut[1] = convTools::noCopyIplImage2Mat(&imageOut2);
            m_tempImageOut[2] = convTools::noCopyIplImage2Mat(&imageOut3);

            std::vector<cv::cuda::GpuMat> mats;
            cv::cuda::split(src, mats);
            mats[0].download(m_tempImageOut[0]);
            mats[1].download(m_tempImageOut[1]);
            mats[2].download(m_tempImageOut[2]);

            if (static_cast<void*>(m_tempImageOut[0].data) != static_cast<void*>(imageOut1.imageData) ||
                static_cast<void*>(m_tempImageOut[1].data) != static_cast<void*>(imageOut2.imageData) ||
                static_cast<void*>(m_tempImageOut[2].data) != static_cast<void*>(imageOut3.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
                Error("cv::Mat data ptr and imageOut data ptr are different.");
        }

        outGuard1.Timestamp() = ts;
        outGuard2.Timestamp() = ts;
        outGuard3.Timestamp() = ts;
    }
    catch (const std::exception& e)
    {
        Error(e.what());
    }
}
