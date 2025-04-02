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
// Purpose of this module : This component can perform rotations and/or flip operations on the input image (+90deg, -90deg, 180 deg, horizontal flip, vertical flip).
////////////////////////////////

#include "maps_OpenCV_RotateAndFlip.h"	// Includes the header of this component

#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

// Use the macros to declare the inputs
MAPS_BEGIN_INPUTS_DEFINITION(MAPSOpenCV_RotateAndFlip)
    MAPS_INPUT("imageIn", MAPS::FilterIplImage, MAPS::FifoReader)
    MAPS_INPUT("angle_in", MAPS::FilterInteger32, MAPS::SamplingReader)
    MAPS_INPUT("i_gpu", Filter_MapsCudaStruct, MAPS::FifoReader)
    MAPS_END_INPUTS_DEFINITION

// Use the macros to declare the outputs
MAPS_BEGIN_OUTPUTS_DEFINITION(MAPSOpenCV_RotateAndFlip)
    MAPS_OUTPUT("imageOut", MAPS::IplImage, nullptr, nullptr, 0)
    MAPS_OUTPUT_USER_DYNAMIC_STRUCTURE("o_gpu", MapsCudaStruct)
MAPS_END_OUTPUTS_DEFINITION

// Use the macros to declare the properties
MAPS_BEGIN_PROPERTIES_DEFINITION(MAPSOpenCV_RotateAndFlip)
    MAPS_PROPERTY_ENUM("operation", "None|90 deg clockwise|90 deg counter-clockwise|180 deg|Flip up-down|Flip left-right|Specify in degrees", 0, false, false)
    MAPS_PROPERTY("use_cuda", false, false, false)
    MAPS_PROPERTY("gpu_mat_as_input", false, false, false)
    MAPS_PROPERTY("gpu_mat_as_output", false, false, false)
    MAPS_PROPERTY_ENUM("angle_input_mode", "Property|Input", 0, false, false)
    MAPS_PROPERTY("angle", 0, false, true)
    MAPS_END_PROPERTIES_DEFINITION

// Use the macros to declare the actions
MAPS_BEGIN_ACTIONS_DEFINITION(MAPSOpenCV_RotateAndFlip)
    //MAPS_ACTION("aName",MAPSOpenCV_RotateAndFlip::ActionName)
MAPS_END_ACTIONS_DEFINITION

//Version 1.1: added rotation with certain angle, eventually provided on input.
//Version 1.2: corrected rotation for 90 deg counter clockwise.

// Use the macros to declare this component (OpenCV_RotateAndFlip) behaviour
MAPS_COMPONENT_DEFINITION(MAPSOpenCV_RotateAndFlip, "OpenCV_RotateAndFlip_cuda", "1.1.0", 128,
                         MAPS::Threaded, MAPS::Threaded,
                         0, // Nb of inputs. Leave -1 to use the number of declared input definitions
                         0, // Nb of outputs. Leave -1 to use the number of declared output definitions
                         2, // Nb of properties. Leave -1 to use the number of declared property definitions
                        -1) // Nb of actions. Leave -1 to use the number of declared action definitions

enum Operation : uint8_t
{
    Operation_None,
    Operation_Rotation_90_ClockWise,
    Operation_Rotation_90_CounterClockWise,
    Operation_Rotation_180,
    Operation_Flip_Up_Down,
    Operation_Flip_Left_Right,
    Operation_Rotation_SpecifiedDegrees
};

void MAPSOpenCV_RotateAndFlip::Dynamic()
{
    m_operation = static_cast<int>(GetIntegerProperty("operation"));
    if (m_operation == 6)
    {
        NewProperty("angle_input_mode");
        m_angleInputMode = static_cast<int>(GetIntegerProperty("angle_input_mode"));

        if (m_angleInputMode == 0)
            NewProperty("angle");
        else
            NewInput("angle_in");
    }

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

void MAPSOpenCV_RotateAndFlip::Birth()
{
    m_inputs.push_back(&Input(0));
    if (m_operation == 6 && m_angleInputMode != 0)
        m_inputs.push_back(&Input(1));

    if (m_useCuda && m_gpuMatAsInput)
    {
        m_inputReader = MAPS::MakeInputReader::Triggered(
            this,
            Input(0),
            MAPS::InputReaderOption::Triggered::TriggerKind::DataInput,
            MAPS::InputReaderOption::Triggered::SamplingBehavior::WaitForAllInputs,
            m_inputs,
            &MAPSOpenCV_RotateAndFlip::AllocateOutputBufferSizeGpu,
            &MAPSOpenCV_RotateAndFlip::ProcessDataGpu
        );
    }
    else
    {
        m_inputReader = MAPS::MakeInputReader::Triggered(
            this,
            Input(0),
            MAPS::InputReaderOption::Triggered::TriggerKind::DataInput,
            MAPS::InputReaderOption::Triggered::SamplingBehavior::WaitForAllInputs,
            m_inputs,
            &MAPSOpenCV_RotateAndFlip::AllocateOutputBufferSize,
            &MAPSOpenCV_RotateAndFlip::ProcessData
        );
    }
}

void MAPSOpenCV_RotateAndFlip::Core()
{
    m_inputReader->Read();
}

void MAPSOpenCV_RotateAndFlip::Death()
{
    m_inputReader.reset();
    m_inputs.clear();
}

void MAPSOpenCV_RotateAndFlip::FreeBuffers()
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

void MAPSOpenCV_RotateAndFlip::AllocateOutputBufferSize(const MAPSTimestamp, const MAPS::ArrayView <MAPS::InputElt<>> inElts)
{
    const IplImage& imageIn = inElts[0].DataAs<IplImage>();
    cv::Mat tempImageIn = convTools::noCopyIplImage2Mat(&imageIn);
    IplImage model;

    switch (m_operation)
    {
    case Operation_None: // None
    case Operation_Rotation_180: // 180 deg
    case Operation_Flip_Up_Down: // Flip up-down
    case Operation_Flip_Left_Right: // Flip left-right
    case Operation_Rotation_SpecifiedDegrees: // Specify in degrees
        model = MAPS::IplImageModel(imageIn.width, imageIn.height, imageIn.channelSeq, imageIn.dataOrder, imageIn.depth, imageIn.align);
        break;

    case Operation_Rotation_90_ClockWise: // 90 deg clockwise
    case Operation_Rotation_90_CounterClockWise: // 90 deg counter-clockwise
        model = MAPS::IplImageModel(imageIn.height, imageIn.width, imageIn.channelSeq, imageIn.dataOrder, imageIn.depth, imageIn.align);
        break;

    default:
        Error("Unknown operation.");
    }

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

void MAPSOpenCV_RotateAndFlip::AllocateOutputBufferSizeGpu(const MAPSTimestamp, const MAPS::ArrayView<MAPS::InputElt<>> inElts)
{
    const IplImage& proxy = inElts[0].DataAs<MapsCudaStruct>().m_IplImageProxy;
    IplImage model;

    switch (m_operation)
    {
    case Operation_None: // None
    case Operation_Rotation_180: // 180 deg
    case Operation_Flip_Up_Down: // Flip up-down
    case Operation_Flip_Left_Right: // Flip left-right
    case Operation_Rotation_SpecifiedDegrees: // Specify in degrees
        model = MAPS::IplImageModel(proxy.width, proxy.height, proxy.channelSeq, proxy.dataOrder, proxy.depth, proxy.align);
        break;

    case Operation_Rotation_90_ClockWise: // 90 deg clockwise
    case Operation_Rotation_90_CounterClockWise: // 90 deg counter-clockwise
        model = MAPS::IplImageModel(proxy.height, proxy.width, proxy.channelSeq, proxy.dataOrder, proxy.depth, proxy.align);
        break;

    default:
        Error("Unknown operation.");
    }

    if (m_gpuMatAsOutput)
    {
        try
        {
            AllocateDynamicOutputBuffers(
                DynamicOutput<MapsCudaStruct>(Output("o_gpu"),
                    [&] { return new MapsCudaStruct(model.width, model.height, model.nChannels, model); }
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

void MAPSOpenCV_RotateAndFlip::ProcessData(const MAPSTimestamp ts, const MAPS::ArrayView <MAPS::InputElt<>> inElts)
{
    MAPS::OutputGuard<> outGuard{ this, Output(0) };
    const IplImage& imageIn = inElts[0].DataAs<IplImage>();
    cv::Mat tempImageIn = convTools::noCopyIplImage2Mat(&imageIn);

    try
    {
        switch (m_operation)
        {
        case Operation_None: // None
            if (m_gpuMatAsOutput)
            {
                MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
                const IplImage& proxyDst = outputData.m_IplImageProxy;
                cv::cuda::GpuMat dst(proxyDst.height, proxyDst.width, CV_MAKETYPE(proxyDst.depth == 8 ? 0 : proxyDst.depth / 8, proxyDst.nChannels), outputData.m_points);
                dst.upload(tempImageIn);
            }
            else
            {
                IplImage& imageOut = outGuard.DataAs<IplImage>();
                memcpy(imageOut.imageData, imageIn.imageData, imageIn.imageSize);
            }
            break;

        case Operation_Rotation_90_ClockWise: // 90 deg clockwise
            Rotate(-90, outGuard, tempImageIn);
            break;

        case Operation_Rotation_90_CounterClockWise: // 90 deg counter-clockwise
            Rotate(90, outGuard, tempImageIn);
            break;

        case Operation_Rotation_180: // 180 deg
            Rotate(180, outGuard, tempImageIn);
            break;

        case Operation_Flip_Up_Down: // Flip up-down
            Flip(0, outGuard, tempImageIn);
            break;

        case Operation_Flip_Left_Right: // Flip left-right
            Flip(1, outGuard, tempImageIn);
            break;

        case Operation_Rotation_SpecifiedDegrees: // Specify in degrees
        {
            int rotationDegrees = 0;
            if (m_angleInputMode == 0)
            {
                rotationDegrees = static_cast<int>(GetIntegerProperty("angle"));
            }
            else if (DataAvailableInFIFO(Input(1)))
            {
                MAPSIOElt* ioeltRot = StartReading(Input(1));
                rotationDegrees = static_cast<int>(ioeltRot->Integer32());
            }

            Rotate(rotationDegrees, outGuard, tempImageIn);
        }
        break;

        default:
            Error("Unknown operation.");
        }
    }
    catch (const std::exception& e)
    {
        Error(e.what());
    }

    outGuard.VectorSize() = 0;
    outGuard.Timestamp() = ts;
}

void MAPSOpenCV_RotateAndFlip::ProcessDataGpu(const MAPSTimestamp ts, const MAPS::ArrayView<MAPS::InputElt<>> inElts)
{
    MAPS::OutputGuard<> outGuard{ this, Output(0) };
    const MapsCudaStruct& imageIn = inElts[0].DataAs<MapsCudaStruct>();
    const IplImage& proxy = imageIn.m_IplImageProxy;
    const cv::cuda::GpuMat src(proxy.height, proxy.width, CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels), imageIn.m_points);

    try
    {
        switch (m_operation)
        {
        case Operation_None: // None
            if (m_gpuMatAsOutput)
            {
                MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
                const IplImage& proxyDst = outputData.m_IplImageProxy;
                cv::cuda::GpuMat dst(proxyDst.height, proxyDst.width, CV_MAKETYPE(proxyDst.depth == 8 ? 0 : proxyDst.depth / 8, proxyDst.nChannels), outputData.m_points);
                src.copyTo(dst);
            }
            else
            {
                IplImage& imageOut = outGuard.DataAs<IplImage>();
                cv::Mat tempImageOut = convTools::noCopyIplImage2Mat(&imageOut);
                src.download(tempImageOut);
            }
            break;

        case Operation_Rotation_90_ClockWise: // 90 deg clockwise
            RotateGpu(-90, outGuard, src);
            break;

        case Operation_Rotation_90_CounterClockWise: // 90 deg counter-clockwise
            RotateGpu(90, outGuard, src);
            break;

        case Operation_Rotation_180: // 180 deg
            RotateGpu(180, outGuard, src);
            break;

        case Operation_Flip_Up_Down: // Flip up-down
            FlipGpu(0, outGuard, src);
            break;

        case Operation_Flip_Left_Right: // Flip left-right
            FlipGpu(1, outGuard, src);
            break;

        case Operation_Rotation_SpecifiedDegrees: // Specify in degrees
        {
            int rotationDegrees = 0;
            if (m_angleInputMode == 0)
            {
                rotationDegrees = static_cast<int>(GetIntegerProperty("angle"));
            }
            else if (DataAvailableInFIFO(Input(1)))
            {
                MAPSIOElt* ioeltRot = StartReading(Input(1));
                rotationDegrees = static_cast<int>(ioeltRot->Integer32());
            }

            RotateGpu(rotationDegrees, outGuard, src);
        }
        break;

        default:
            Error("Unknown operation.");
        }
    }
    catch (const std::exception& e)
    {
        Error(e.what());
    }

    outGuard.VectorSize() = 0;
    outGuard.Timestamp() = ts;
}

void MAPSOpenCV_RotateAndFlip::Rotate(int degrees, MAPS::OutputGuard<>& outGuard, const cv::Mat& imageIn)
{
    cv::Mat rotationMatrix = cv::Mat(2, 3, CV_32FC1);
    cv::Point2f center;
    center.x = imageIn.cols / 2.0f;
    center.y = imageIn.rows / 2.0f;

    if (m_useCuda)
    {
        rotationMatrix = cv::getRotationMatrix2D(center, degrees, 1.0);
        cv::cuda::GpuMat src(imageIn);

        if (m_gpuMatAsOutput)
        {
            MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
            const IplImage& proxyDst = outputData.m_IplImageProxy;
            cv::cuda::GpuMat dst(proxyDst.height, proxyDst.width, CV_MAKETYPE(proxyDst.depth == 8 ? 0 : proxyDst.depth / 8, proxyDst.nChannels), outputData.m_points);
            cv::cuda::warpAffine(src, dst, rotationMatrix, imageIn.size());
        }
        else
        {
            IplImage& imageOut = outGuard.DataAs<IplImage>();
            cv::Mat tempImageOut = convTools::noCopyIplImage2Mat(&imageOut);
            cv::cuda::GpuMat dst;
            cv::cuda::warpAffine(src, dst, rotationMatrix, imageIn.size());
            dst.download(tempImageOut);
        }
    }
    else
    {
        IplImage& imageOut = outGuard.DataAs<IplImage>();
        cv::Mat tempImageOut = convTools::noCopyIplImage2Mat(&imageOut);

        rotationMatrix = cv::getRotationMatrix2D(center, degrees, 1.0);
        cv::warpAffine(imageIn, tempImageOut, rotationMatrix, imageIn.size());

        if (static_cast<void*>(tempImageOut.data) != static_cast<void*>(imageOut.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
            Error("cv::Mat data ptr and imageOut data ptr are different.");
    }
}

void MAPSOpenCV_RotateAndFlip::RotateGpu(int degrees, MAPS::OutputGuard<>& outGuard, const cv::cuda::GpuMat& imageIn)
{
    cv::Mat rotationMatrix = cv::Mat(2, 3, CV_32FC1);
    cv::Point2f center;
    center.x = imageIn.cols / 2.0f;
    center.y = imageIn.rows / 2.0f;
    rotationMatrix = cv::getRotationMatrix2D(center, degrees, 1.0);

    if (m_gpuMatAsOutput)
    {
        MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
        const IplImage& proxyDst = outputData.m_IplImageProxy;
        cv::cuda::GpuMat dst(proxyDst.height, proxyDst.width, CV_MAKETYPE(proxyDst.depth == 8 ? 0 : proxyDst.depth / 8, proxyDst.nChannels), outputData.m_points);
        cv::cuda::warpAffine(imageIn, dst, rotationMatrix, imageIn.size());
    }
    else
    {
        IplImage& imageOut = outGuard.DataAs<IplImage>();
        cv::Mat tempImageOut = convTools::noCopyIplImage2Mat(&imageOut);
        cv::cuda::GpuMat dst;
        cv::cuda::warpAffine(imageIn, dst, rotationMatrix, imageIn.size());
        dst.download(tempImageOut);
    }
}

void MAPSOpenCV_RotateAndFlip::Flip(int flipMode, MAPS::OutputGuard<>& outGuard, const cv::Mat& imageIn)
{
    if (m_useCuda)
    {
        cv::cuda::GpuMat src(imageIn);

        if (m_gpuMatAsOutput)
        {
            MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
            const IplImage& proxyDst = outputData.m_IplImageProxy;
            cv::cuda::GpuMat dst(proxyDst.height, proxyDst.width, CV_MAKETYPE(proxyDst.depth == 8 ? 0 : proxyDst.depth / 8, proxyDst.nChannels), outputData.m_points);
            cv::cuda::flip(src, dst, flipMode);
        }
        else
        {
            IplImage& imageOut = outGuard.DataAs<IplImage>();
            cv::Mat tempImageOut = convTools::noCopyIplImage2Mat(&imageOut);
            cv::cuda::GpuMat dst;
            cv::cuda::flip(src, dst, flipMode);
            dst.download(tempImageOut);
        }
    }
    else
    {
        IplImage& imageOut = outGuard.DataAs<IplImage>();
        cv::Mat tempImageOut = convTools::noCopyIplImage2Mat(&imageOut);

        cv::flip(imageIn, tempImageOut, flipMode);

        if (static_cast<void*>(tempImageOut.data) != static_cast<void*>(imageOut.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
            Error("cv::Mat data ptr and imageOut data ptr are different.");
    }
}

void MAPSOpenCV_RotateAndFlip::FlipGpu(int flipMode, MAPS::OutputGuard<>& outGuard, const cv::cuda::GpuMat& imageIn)
{
    if (m_gpuMatAsOutput)
    {
        MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
        const IplImage& proxyDst = outputData.m_IplImageProxy;
        cv::cuda::GpuMat dst(proxyDst.height, proxyDst.width, CV_MAKETYPE(proxyDst.depth == 8 ? 0 : proxyDst.depth / 8, proxyDst.nChannels), outputData.m_points);
        cv::cuda::flip(imageIn, dst, flipMode);
    }
    else
    {
        IplImage& imageOut = outGuard.DataAs<IplImage>();
        cv::Mat tempImageOut = convTools::noCopyIplImage2Mat(&imageOut);
        cv::cuda::GpuMat dst;
        cv::cuda::flip(imageIn, dst, flipMode);
        dst.download(tempImageOut);
    }
}
