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
// Purpose of this module : Take an image in input, resize and output it.
////////////////////////////////

#include "maps_OpenCV_Resize.h"	// Includes the header of this component
#include "maps_io_access.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudawarping.hpp>

// Use the macros to declare the inputs
MAPS_BEGIN_INPUTS_DEFINITION(MAPSOpenCV_Resize)
MAPS_INPUT("imageIn", MAPS::FilterIplImage, MAPS::FifoReader)
MAPS_INPUT("i_gpu", Filter_MapsCudaStruct, MAPS::FifoReader)
MAPS_END_INPUTS_DEFINITION

// Use the macros to declare the outputs
MAPS_BEGIN_OUTPUTS_DEFINITION(MAPSOpenCV_Resize)
MAPS_OUTPUT("imageOut", MAPS::IplImage, nullptr, nullptr, 0)
MAPS_OUTPUT_USER_DYNAMIC_STRUCTURE("o_gpu", MapsCudaStruct)
MAPS_END_OUTPUTS_DEFINITION

// Use the macros to declare the properties
MAPS_BEGIN_PROPERTIES_DEFINITION(MAPSOpenCV_Resize)
MAPS_PROPERTY("new_size_x", 320, false, false)
MAPS_PROPERTY("new_size_y", 240, false, false)
MAPS_PROPERTY_ENUM("interpolation", "Nearest Neighbor|Bilinear|Bicubic|Area|Lanczos|Linear Exact", 1, false, true)
MAPS_PROPERTY("use_cuda", false, false, false)
MAPS_PROPERTY("gpu_mat_as_input", false, false, false)
MAPS_PROPERTY("gpu_mat_as_output", false, false, false)
MAPS_END_PROPERTIES_DEFINITION

// Use the macros to declare the actions
MAPS_BEGIN_ACTIONS_DEFINITION(MAPSOpenCV_Resize)
    //MAPS_ACTION("aName",MAPSOpenCV_Resize::ActionName)
MAPS_END_ACTIONS_DEFINITION

// Use the macros to declare this component (OpenCV_Resize) behaviour
MAPS_COMPONENT_DEFINITION(MAPSOpenCV_Resize, "OpenCV_Resize_cuda", "1.1.0", 128,
                            MAPS::Threaded | MAPS::Sequential, MAPS::Threaded,
                            0, // Nb of inputs
                            0, // Nb of outputs
                            4, // Nb of properties
                            -1) // Nb of actions

void MAPSOpenCV_Resize::Birth()
{
    m_newSize = cv::Size(static_cast<int>(GetIntegerProperty("new_size_x")), static_cast<int>(GetIntegerProperty("new_size_y")));
    UpdateInterp(GetIntegerProperty("interpolation"));
   
    if (m_useCuda && m_gpuMatAsInput)
    {
        m_inputReader = MAPS::MakeInputReader::Reactive(
            this,
            Input(0),
            &MAPSOpenCV_Resize::AllocateOutputBufferSizeGpu,  // Called when data is received for the first time only
            &MAPSOpenCV_Resize::ProcessDataGpu      // Called when data is received for the first time AND all subsequent times
        );
    }
    else
    {
        m_inputReader = MAPS::MakeInputReader::Reactive(
            this,
            Input(0),
            &MAPSOpenCV_Resize::AllocateOutputBufferSize,  // Called when data is received for the first time only
            &MAPSOpenCV_Resize::ProcessData      // Called when data is received for the first time AND all subsequent times
        );
    }

}

void MAPSOpenCV_Resize::Core()
{
    m_inputReader->Read();
}

void MAPSOpenCV_Resize::Death()
{
    m_inputReader.reset();
}

void MAPSOpenCV_Resize::AllocateOutputBufferSize(const MAPSTimestamp, const MAPS::InputElt<IplImage> imageInElt)
{
    const IplImage& imageIn = imageInElt.Data();
    IplImage model = MAPS::IplImageModel(m_newSize.width, m_newSize.height, imageIn.channelSeq, imageIn.dataOrder, imageIn.depth, imageIn.align);

    if (m_gpuMatAsOutput)
    {
        try
        {
            AllocateDynamicOutputBuffers(
                DynamicOutput<MapsCudaStruct>(Output("o_gpu"),
                    [&] { return new MapsCudaStruct(model.width, model.height, model.nChannels, model); }  // struct allocation
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
        Output(0).AllocOutputBufferIplImage(model);
    }
}

void MAPSOpenCV_Resize::ProcessData(const MAPSTimestamp ts, const MAPS::InputElt<IplImage> inElt)
{
    try
    {
        MAPS::OutputGuard<> outGuard{ this, Output(0) };
        cv::Mat tempImageIn = convTools::noCopyIplImage2Mat(&inElt.Data());

        if (m_useCuda)
        {
            cv::cuda::GpuMat src(tempImageIn);
            if (m_gpuMatAsOutput)
            {
                MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
                const IplImage& proxyDst = outputData.m_IplImageProxy;
                cv::cuda::GpuMat dst(proxyDst.height, proxyDst.width, CV_MAKETYPE(proxyDst.depth == 8 ? 0 : proxyDst.depth / 8, proxyDst.nChannels), outputData.m_points);
                cv::cuda::resize(src, dst, m_newSize, 0, 0, m_method);
            }
            else
            {
                const IplImage& imageOut = outGuard.DataAs<IplImage>();
                cv::Mat tempImageOut = convTools::noCopyIplImage2Mat(&imageOut);
                cv::cuda::GpuMat dst;
                cv::cuda::resize(src, dst, m_newSize, 0, 0, m_method);
                dst.download(tempImageOut);

                if (static_cast<void*>(tempImageOut.data) != static_cast<void*>(imageOut.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
                    Error("cv::Mat data ptr and imageOut data ptr are different.");
            }
        }
        else
        {
            const IplImage& imageOut = outGuard.DataAs<IplImage>();
            cv::Mat tempImageOut = convTools::noCopyIplImage2Mat(&imageOut);
            cv::resize(tempImageIn, tempImageOut, m_newSize, 0, 0, m_method);

            if (static_cast<void*>(tempImageOut.data) != static_cast<void*>(imageOut.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
                Error("cv::Mat data ptr and imageOut data ptr are different.");
        }

        outGuard.Timestamp() = ts;
    }
    catch (const std::exception& e)
    {
        Error(e.what());
    }
}

void MAPSOpenCV_Resize::AllocateOutputBufferSizeGpu(const MAPSTimestamp, const MAPS::InputElt<MapsCudaStruct> imageInElt)
{
    const MapsCudaStruct& imageIn = imageInElt.Data();

    if (m_gpuMatAsOutput)
    {
        try
        {
            AllocateDynamicOutputBuffers(
                DynamicOutput<MapsCudaStruct>(Output("o_gpu"),
                    [&] { return new MapsCudaStruct(m_newSize.width, m_newSize.height, imageIn.m_IplImageProxy.nChannels, imageIn.m_IplImageProxy); }
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
        const IplImage& proxy = imageIn.m_IplImageProxy;
        IplImage model = MAPS::IplImageModel(m_newSize.width, m_newSize.height, proxy.channelSeq, proxy.dataOrder, proxy.depth, proxy.align);
        Output(0).AllocOutputBufferIplImage(model);
    }  
}

void MAPSOpenCV_Resize::ProcessDataGpu(const MAPSTimestamp ts, const MAPS::InputElt<MapsCudaStruct> inElt)
{
    try
    {
        MAPS::OutputGuard<> outGuard{ this, Output(0) };
        const IplImage& proxy = inElt.Data().m_IplImageProxy;
        const cv::cuda::GpuMat src(proxy.height, proxy.width, CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels), inElt.Data().m_points);

        if (m_gpuMatAsOutput)
        {
            MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
            const IplImage& proxyDst = outputData.m_IplImageProxy;
            cv::cuda::GpuMat dst(proxyDst.height, proxyDst.width, CV_MAKETYPE(proxyDst.depth == 8 ? 0 : proxyDst.depth / 8, proxyDst.nChannels), outputData.m_points);
            cv::cuda::resize(src, dst, m_newSize, 0, 0, m_method);
            outGuard.Timestamp() = ts;
        }
        else
        {
            IplImage& imageOut = outGuard.DataAs<IplImage>();
            cv::Mat tempImageOut = convTools::noCopyIplImage2Mat(&imageOut);
            cv::cuda::GpuMat dst;
            cv::cuda::resize(src, dst, m_newSize, 0, 0, m_method);
            dst.download(tempImageOut);
            outGuard.Timestamp() = ts;

            if (static_cast<void*>(tempImageOut.data) != static_cast<void*>(imageOut.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
                Error("cv::Mat data ptr and imageOut data ptr are different.");
        }
    }
    catch (const std::exception& e)
    {
        Error(e.what());
    }
}

void MAPSOpenCV_Resize::UpdateInterp(MAPSInt64 selectedEnum)
{
    switch (selectedEnum)
    {
        case 0: // Nearest Neighbor
            m_method = cv::INTER_NEAREST;
            break;
        case 1: // Bilinear
            m_method = cv::INTER_LINEAR;
            break;
        case 2: // Bicubic
            m_method = cv::INTER_CUBIC;
            break;
        case 3: // Area
            m_method = cv::INTER_AREA;
            break;
        case 4: // Lanczos
            m_method = cv::INTER_LANCZOS4;
            break;
        case 5: // Linear Exact
            m_method = cv::INTER_LINEAR_EXACT;
            break;
        default:
            Error("Unknown interpolation method.");
    }
}

void MAPSOpenCV_Resize::Set(MAPSProperty& p, MAPSInt64 value)
{
    MAPSComponent::Set(p, value);
    if (p.ShortName() == "interpolation")
    {
        UpdateInterp(value);
    }
}

void MAPSOpenCV_Resize::Set(MAPSProperty& p, const MAPSString& value)
{
    MAPSComponent::Set(p, value);
    if (p.ShortName() == "interpolation")
    {
        UpdateInterp(GetEnumProperty("interpolation").selectedEnum);
    }
}

void MAPSOpenCV_Resize::Dynamic()
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

void MAPSOpenCV_Resize::FreeBuffers()
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

void MAPSOpenCV_Resize::Set(MAPSProperty& p, const MAPSEnumStruct& enumStruct)
{
    MAPSComponent::Set(p, enumStruct);
    if (p.ShortName() == "interpolation")
    {
        UpdateInterp(enumStruct.selectedEnum);
    }
}