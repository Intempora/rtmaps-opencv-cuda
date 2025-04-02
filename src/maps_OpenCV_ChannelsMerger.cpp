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
// Purpose of this module : Composes multi-channel array from several single channel arrays or inserts a single channel into the array.
////////////////////////////////

#include "maps_OpenCV_ChannelsMerger.h"	// Includes the header of this component

#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudawarping.hpp>
#include "opencv2/cudaarithm.hpp"

// Use the macros to declare the inputs
MAPS_BEGIN_INPUTS_DEFINITION(MAPSOpenCV_ChannelsMerger)
    MAPS_INPUT("channel1", MAPS::FilterIplImage, MAPS::FifoReader)
    MAPS_INPUT("channel2", MAPS::FilterIplImage, MAPS::FifoReader)
    MAPS_INPUT("channel3", MAPS::FilterIplImage, MAPS::FifoReader)
    MAPS_INPUT("i_gpu_channel1", Filter_MapsCudaStruct, MAPS::FifoReader)
    MAPS_INPUT("i_gpu_channel2", Filter_MapsCudaStruct, MAPS::FifoReader)
    MAPS_INPUT("i_gpu_channel3", Filter_MapsCudaStruct, MAPS::FifoReader)
    MAPS_END_INPUTS_DEFINITION

// Use the macros to declare the outputs
MAPS_BEGIN_OUTPUTS_DEFINITION(MAPSOpenCV_ChannelsMerger)
    MAPS_OUTPUT("imageOut", MAPS::IplImage, nullptr, nullptr, 0)
    MAPS_OUTPUT_USER_DYNAMIC_STRUCTURE("o_gpu", MapsCudaStruct)
    MAPS_END_OUTPUTS_DEFINITION

// Use the macros to declare the properties
MAPS_BEGIN_PROPERTIES_DEFINITION(MAPSOpenCV_ChannelsMerger)
    MAPS_PROPERTY("outputChannelSeq", "BGR", false, false)
    MAPS_PROPERTY("outputPlanar", false, false, false)
    MAPS_PROPERTY("synchro_tolerance", 0, false, false)
    MAPS_PROPERTY("use_cuda", false, false, false)
    MAPS_PROPERTY("gpu_mat_as_input", false, false, false)
    MAPS_PROPERTY("gpu_mat_as_output", false, false, false)
MAPS_END_PROPERTIES_DEFINITION

// Use the macros to declare the actions
MAPS_BEGIN_ACTIONS_DEFINITION(MAPSOpenCV_ChannelsMerger)
    //MAPS_ACTION("aName",MAPSOpenCV_ChannelsMerger::ActionName)
MAPS_END_ACTIONS_DEFINITION

// Use the macros to declare this component (OpenCV_Resize) behaviour
MAPS_COMPONENT_DEFINITION(MAPSOpenCV_ChannelsMerger, "OpenCV_ChannelsMerger_cuda", "1.1.0", 128,
                            MAPS::Threaded|MAPS::Sequential, MAPS::Threaded,
                            0, // Nb of inputs
                            0, // Nb of outputs
                            4, // Nb of properties
                            -1) // Nb of actions


void MAPSOpenCV_ChannelsMerger::Birth()
{
    m_isOutputPlanar = GetBoolProperty("outputPlanar");
    m_channelSeq = GetStringProperty("outputChannelSeq");
    m_tempGpuMats.resize(3);

    if (m_channelSeq.size() > 4)
        Error("outputChannelSeq property : Channel sequence is too long. It must be made of 4 characters max. (ex : RGB, BGR, YUV, HSV, etc...)");

    if (m_useCuda && m_gpuMatAsInput)
    {
        m_inputReader = MAPS::MakeInputReader::Synchronized(
            this,
            GetIntegerProperty("synchro_tolerance"),
            MAPS::InputReaderOption::Synchronized::SyncBehavior::SyncAllInputs,
            MAPS::MakeArray(&Input(0), &Input(1), &Input(2)),
            &MAPSOpenCV_ChannelsMerger::AllocateOutputBufferSizeGpu,
            &MAPSOpenCV_ChannelsMerger::ProcessDataGpu
        );
    }
    else
    {
        m_inputReader = MAPS::MakeInputReader::Synchronized(
            this,
            GetIntegerProperty("synchro_tolerance"),
            MAPS::InputReaderOption::Synchronized::SyncBehavior::SyncAllInputs,
            MAPS::MakeArray(&Input(0), &Input(1), &Input(2)),
            &MAPSOpenCV_ChannelsMerger::AllocateOutputBufferSize,
            &MAPSOpenCV_ChannelsMerger::ProcessData
        );
    }
}

void MAPSOpenCV_ChannelsMerger::Core()
{
    m_inputReader->Read();
}

void MAPSOpenCV_ChannelsMerger::Death()
{
    m_inputReader.reset();
}

void MAPSOpenCV_ChannelsMerger::Dynamic()
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
            NewInput("i_gpu_channel1");
            NewInput("i_gpu_channel2");
            NewInput("i_gpu_channel3");
        }
        else
        {
            NewInput("channel1");
            NewInput("channel2");
            NewInput("channel3");
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
        NewInput("channel1");
        NewInput("channel2");
        NewInput("channel3");
        NewOutput("imageOut");
    }
}

void MAPSOpenCV_ChannelsMerger::FreeBuffers()
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

void MAPSOpenCV_ChannelsMerger::AllocateOutputBufferSize(const MAPSTimestamp, const MAPS::ArrayView<MAPS::InputElt<IplImage>> imageInElts)
{
    const IplImage& imageIn1 = imageInElts[0].Data();

    m_tempImageIn[0] = convTools::noCopyIplImage2Mat(&imageIn1); // Convert IplImage to cv::Mat
    m_tempImageIn[1] = convTools::noCopyIplImage2Mat(&imageInElts[1].Data()); // Convert IplImage to cv::Mat
    m_tempImageIn[2] = convTools::noCopyIplImage2Mat(&imageInElts[2].Data()); // Convert IplImage to cv::Mat

    if (m_tempImageIn[0].channels() != 1)
        Error("Input 1 : This component only supports single channel images on its inputs.");

    if (m_tempImageIn[1].channels() != 1)
        Error("Input 2 : This component only supports single channel images on its inputs.");

    if (m_tempImageIn[2].channels() != 1)
        Error("Input 3 : This component only supports single channel images on its inputs.");

    if (m_tempImageIn[0].size() != m_tempImageIn[1].size() ||
        m_tempImageIn[0].size() != m_tempImageIn[2].size())
        Error("Input images must have the same dimensions.");

    IplImage model = MAPS::IplImageModel(imageIn1.width, imageIn1.height, m_channelSeq.c_str(), m_isOutputPlanar ? IPL_DATA_ORDER_PLANE : IPL_DATA_ORDER_PIXEL, imageIn1.depth, imageIn1.align);

    if (m_gpuMatAsOutput)
    {
        try
        {
            AllocateDynamicOutputBuffers(
                DynamicOutput<MapsCudaStruct>(Output("o_gpu"),
                    [&] { return new MapsCudaStruct(m_tempImageIn[0].cols, m_tempImageIn[0].rows, m_tempImageIn[0].channels(), model); }  // struct allocation
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
        Output("imageOut").AllocOutputBufferIplImage(model);
    }
}

void MAPSOpenCV_ChannelsMerger::ProcessData(const MAPSTimestamp ts, const MAPS::ArrayView<MAPS::InputElt<IplImage>> inElts)
{
    try
    {
        const IplImage& imageIn1 = inElts[0].Data();
        const IplImage& imageIn2 = inElts[1].Data();
        const IplImage& imageIn3 = inElts[2].Data();
        m_tempImageIn[0] = convTools::noCopyIplImage2Mat(&imageIn1); // Convert IplImage to cv::Mat
        m_tempImageIn[1] = convTools::noCopyIplImage2Mat(&imageIn2); // Convert IplImage to cv::Mat
        m_tempImageIn[2] = convTools::noCopyIplImage2Mat(&imageIn3); // Convert IplImage to cv::Mat
        MAPS::OutputGuard<> outGuard{ this, Output(0) };

        if (m_useCuda)
        {
            m_tempGpuMats[0].upload(m_tempImageIn[0]);
            m_tempGpuMats[1].upload(m_tempImageIn[1]);
            m_tempGpuMats[2].upload(m_tempImageIn[2]);

            if (m_gpuMatAsOutput)
            {
                MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
                const IplImage& proxy = outputData.m_IplImageProxy;
                cv::cuda::GpuMat dst(proxy.height, proxy.width, CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels), outputData.m_points);

                cv::cuda::merge(m_tempGpuMats, dst);
            }
            else
            {
                IplImage& imageOut = outGuard.DataAs<IplImage>();
                m_tempImageOut = convTools::noCopyIplImage2Mat(&imageOut);
                cv::cuda::GpuMat dst;
                cv::cuda::merge(m_tempGpuMats, dst);
                dst.download(m_tempImageOut);

                if (static_cast<void*>(m_tempImageOut.data) != static_cast<void*>(imageOut.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
                    Error("cv::Mat data ptr and imageOut data ptr are different.");
            }
        }
        else
        {
            IplImage& imageOut = outGuard.DataAs<IplImage>();

            if (m_isOutputPlanar)
            {
                size_t imageSize = imageIn1.imageSize;
                for (size_t i = 0; i < inElts.size(); ++i)
                {
                    std::memcpy(imageOut.imageData + i * imageSize, inElts[i].Data().imageData, imageSize);
                }
            }
            else
            {
                m_tempImageOut = convTools::noCopyIplImage2Mat(&imageOut); // Convert IplImage to cv::Mat without copying

                cv::merge(m_tempImageIn, m_tempImageOut); // Merge the 3 element of m_tempImageIn into m_tempImageOut

                if (static_cast<void*>(m_tempImageOut.data) != static_cast<void*>(imageOut.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
                    Error("cv::Mat data ptr and imageOut data ptr are different.");
            }
        }

        outGuard.Timestamp() = ts;
    }
    catch (const std::exception& e)
    {
        Error(e.what());
    }
}

void MAPSOpenCV_ChannelsMerger::AllocateOutputBufferSizeGpu(const MAPSTimestamp, const MAPS::ArrayView<MAPS::InputElt<MapsCudaStruct>> imageInElts)
{
    const MapsCudaStruct& imageIn1 = imageInElts[0].Data();
    const MapsCudaStruct& imageIn2 = imageInElts[1].Data();
    const MapsCudaStruct& imageIn3 = imageInElts[2].Data();

    if (imageIn1.m_IplImageProxy.nChannels != 1)
        Error("Input 1 : This component only supports single channel images on its inputs.");

    if (imageIn2.m_IplImageProxy.nChannels != 1)
        Error("Input 2 : This component only supports single channel images on its inputs.");

    if (imageIn3.m_IplImageProxy.nChannels != 1)
        Error("Input 3 : This component only supports single channel images on its inputs.");

    if (imageIn1.m_size != imageIn2.m_size ||
        imageIn1.m_size != imageIn3.m_size)
        Error("Input images must have the same dimensions.");

    IplImage model = MAPS::IplImageModel(imageIn1.m_IplImageProxy.width, imageIn1.m_IplImageProxy.height, m_channelSeq.c_str(), m_isOutputPlanar ? IPL_DATA_ORDER_PLANE : IPL_DATA_ORDER_PIXEL,
        imageIn1.m_IplImageProxy.depth, imageIn1.m_IplImageProxy.align);

    int type = CV_MAKETYPE(imageIn1.m_IplImageProxy.depth == 8 ? 0 : imageIn1.m_IplImageProxy.depth / 8, imageIn1.m_IplImageProxy.nChannels);

    if (m_gpuMatAsOutput)
    {
        try
        {
            AllocateDynamicOutputBuffers(
                DynamicOutput<MapsCudaStruct>(Output("o_gpu"),
                    [&] { return new MapsCudaStruct(imageIn1.m_size*3, model); }
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
        Output("imageOut").AllocOutputBufferIplImage(model);
    }
}

void MAPSOpenCV_ChannelsMerger::ProcessDataGpu(const MAPSTimestamp ts, const MAPS::ArrayView<MAPS::InputElt<MapsCudaStruct>> inElts)
{
    try
    {
        const IplImage& proxy = inElts[0].Data().m_IplImageProxy;
        const cv::cuda::GpuMat src1(proxy.height, proxy.width, CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels), inElts[0].Data().m_points);
        const cv::cuda::GpuMat src2(proxy.height, proxy.width, CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels), inElts[1].Data().m_points);
        const cv::cuda::GpuMat src3(proxy.height, proxy.width, CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels), inElts[2].Data().m_points);

        MAPS::OutputGuard<> outGuard{ this, Output(0) };

        m_tempGpuMats[0] = src1;
        m_tempGpuMats[1] = src2;
        m_tempGpuMats[2] = src3;

        if (m_gpuMatAsOutput)
        {
            MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
            const IplImage& proxyDst = outputData.m_IplImageProxy;
            cv::cuda::GpuMat dst(proxyDst.height, proxyDst.width, CV_MAKETYPE(proxyDst.depth == 8 ? 0 : proxyDst.depth / 8, proxyDst.nChannels), outputData.m_points);
            cv::cuda::merge(m_tempGpuMats, dst);
        }
        else
        {
            IplImage& imageOut = outGuard.DataAs<IplImage>();
            m_tempImageOut = convTools::noCopyIplImage2Mat(&imageOut);
            cv::cuda::GpuMat dst;
            cv::cuda::merge(m_tempGpuMats, dst);
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
