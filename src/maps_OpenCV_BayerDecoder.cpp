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
// Purpose of this module : Bayer pattern is widely used in CCD and CMOS cameras. It allows to get color picture
//                          out of a single plane where R, G and B pixels(sensors of a particular component) are interleaved.
////////////////////////////////

#include "maps_OpenCV_BayerDecoder.h"	// Includes the header of this component

#include "opencv2/cudaimgproc.hpp"

// Use the macros to declare the inputs
MAPS_BEGIN_INPUTS_DEFINITION(MAPSBayerDecoder)
MAPS_INPUT("input_ipl", MAPS::FilterIplImage, MAPS::FifoReader)
MAPS_INPUT("input_maps", MAPS::FilterMAPSImage, MAPS::FifoReader)
MAPS_INPUT("i_gpu", Filter_MapsCudaStruct, MAPS::FifoReader)
MAPS_END_INPUTS_DEFINITION

// Use the macros to declare the outputs
MAPS_BEGIN_OUTPUTS_DEFINITION(MAPSBayerDecoder)
MAPS_OUTPUT("imageOut", MAPS::IplImage, nullptr, nullptr, 0)
MAPS_OUTPUT_USER_DYNAMIC_STRUCTURE("o_gpu", MapsCudaStruct)
MAPS_END_OUTPUTS_DEFINITION

// Use the macros to declare the properties
MAPS_BEGIN_PROPERTIES_DEFINITION(MAPSBayerDecoder)
    MAPS_PROPERTY_ENUM("input_type", "IPLImage|MAPSImage", 1, false, true)
    MAPS_PROPERTY_ENUM("input_pattern", "BG|GB|RG|GR", 0, false, true)
    MAPS_PROPERTY_ENUM("outputFormat", "BGR|RGB|BGRA|RGBA", 0, false, false)
    MAPS_PROPERTY("use_cuda", false, false, false)
    MAPS_PROPERTY("gpu_mat_as_input", false, false, false)
    MAPS_PROPERTY("gpu_mat_as_output", false, false, false)
MAPS_END_PROPERTIES_DEFINITION

// Use the macros to declare the actions
MAPS_BEGIN_ACTIONS_DEFINITION(MAPSBayerDecoder)
MAPS_END_ACTIONS_DEFINITION

// Use the macros to declare this component (ColorConvert_Bayer2RGB) behaviour
MAPS_COMPONENT_DEFINITION(MAPSBayerDecoder,"OpenCV_BayerDecoder_cuda", "1.2.2", 128,
                            MAPS::Threaded|MAPS::Sequential, MAPS::Sequential,
                            0, // Nb of inputs
                            0, // Nb of outputs
                            4, // Nb of properties
                            -1) // Nb of actions

enum MAPS_BAYER_PATTERN : uint8_t
{
    MAPS_BAYER_PATTERN_BG,
    MAPS_BAYER_PATTERN_GB,
    MAPS_BAYER_PATTERN_RG,
    MAPS_BAYER_PATTERN_GR
};

void MAPSBayerDecoder::Birth()
{
    m_outputFormat = static_cast<OUTPUT_FORMAT>(GetIntegerProperty("outputFormat"));
    m_pattern = static_cast<MAPS_BAYER_PATTERN>(GetEnumProperty("input_pattern").GetSelected());

    switch (m_outputFormat)
    {
    case OUTPUT_FORMAT::BGR:
        switch (m_pattern)
        {
        case MAPS_BAYER_PATTERN_BG:
            m_colorConvCode = cv::COLOR_BayerBG2BGR;
            break;
        case MAPS_BAYER_PATTERN_GB:
            m_colorConvCode = cv::COLOR_BayerGB2BGR;
            break;
        case MAPS_BAYER_PATTERN_RG:
            m_colorConvCode = cv::COLOR_BayerRG2BGR;
            break;
        case MAPS_BAYER_PATTERN_GR:
            m_colorConvCode = cv::COLOR_BayerGR2BGR;
            break;
        }
        break;
    case OUTPUT_FORMAT::RGB:
        switch (m_pattern)
        {
        case MAPS_BAYER_PATTERN_BG:
            m_colorConvCode = cv::COLOR_BayerBG2RGB;
            break;
        case MAPS_BAYER_PATTERN_GB:
            m_colorConvCode = cv::COLOR_BayerGB2RGB;
            break;
        case MAPS_BAYER_PATTERN_RG:
            m_colorConvCode = cv::COLOR_BayerRG2RGB;
            break;
        case MAPS_BAYER_PATTERN_GR:
            m_colorConvCode = cv::COLOR_BayerGR2RGB;
            break;
        }
        break;
    case OUTPUT_FORMAT::BGRA:
        switch (m_pattern)
        {
        case MAPS_BAYER_PATTERN_BG:
            m_colorConvCode = cv::COLOR_BayerBG2BGRA;
            break;
        case MAPS_BAYER_PATTERN_GB:
            m_colorConvCode = cv::COLOR_BayerGB2BGRA;
            break;
        case MAPS_BAYER_PATTERN_RG:
            m_colorConvCode = cv::COLOR_BayerRG2BGRA;
            break;
        case MAPS_BAYER_PATTERN_GR:
            m_colorConvCode = cv::COLOR_BayerGR2BGRA;
            break;
        }
        break;
    case OUTPUT_FORMAT::RGBA:
        switch (m_pattern)
        {
        case MAPS_BAYER_PATTERN_BG:
            m_colorConvCode = cv::COLOR_BayerBG2RGBA;
            break;
        case MAPS_BAYER_PATTERN_GB:
            m_colorConvCode = cv::COLOR_BayerGB2RGBA;
            break;
        case MAPS_BAYER_PATTERN_RG:
            m_colorConvCode = cv::COLOR_BayerRG2RGBA;
            break;
        case MAPS_BAYER_PATTERN_GR:
            m_colorConvCode = cv::COLOR_BayerGR2RGBA;
            break;
        }
        break;
    }

    if (m_useCuda && m_gpuMatAsInput)
    {
        m_inputReader = MAPS::MakeInputReader::Reactive(
            this,
            Input(0),
            &MAPSBayerDecoder::AllocateOutputBufferGpu,  // Called when data is received for the first time only
            &MAPSBayerDecoder::ProcessDataGpu      // Called when data is received for the first time AND all subsequent times
        );
    }
    else
    {
        if (GetIntegerProperty("input_type") == 0)
        {
            m_inputReader = MAPS::MakeInputReader::Reactive(
                this,
                Input(0),
                &MAPSBayerDecoder::AllocateOutputBufferIpl,  // Called when data is received for the first time only
                &MAPSBayerDecoder::ProcessDataIpl      // Called when data is received for the first time AND all subsequent times
            );
        }
        else
        {
            m_inputReader = MAPS::MakeInputReader::Reactive(
                this,
                Input(0),
                &MAPSBayerDecoder::AllocateOutputBufferMaps,  // Called when data is received for the first time only
                &MAPSBayerDecoder::ProcessDataMaps     // Called when data is received for the first time AND all subsequent times
            );
        }
    }
}

void MAPSBayerDecoder::Dynamic()
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
            if (GetIntegerProperty("input_type") == 0)
            {
                NewInput("input_ipl");
            }
            else
            {
                NewInput("input_maps");
            }
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
        if (GetIntegerProperty("input_type") == 0)
        {
            NewInput("input_ipl");
        }
        else
        {
            NewInput("input_maps");
        }
        NewOutput("imageOut");
    }
}

void MAPSBayerDecoder::FreeBuffers()
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

void MAPSBayerDecoder::Core()
{
    m_inputReader->Read();
}

void MAPSBayerDecoder::Death()
{
    m_inputReader.reset();
}

void MAPSBayerDecoder::Set(MAPSProperty& p, const MAPSString& value)
{
    MAPSComponent::Set(p, value);
    if (p.ShortName() == "input_pattern")
    {
        m_pattern = static_cast<MAPS_BAYER_PATTERN>(GetEnumProperty("input_pattern").GetSelected());
    }
}

void MAPSBayerDecoder::AllocateOutputBufferIpl(const MAPSTimestamp, const MAPS::InputElt<IplImage> imageInElt)
{
    const IplImage& imageIn = imageInElt.Data();

    if (*(MAPSUInt32*)imageIn.channelSeq != MAPS_CHANNELSEQ_GRAY)
        Error("This component only accepts GRAY images on its input (8 bpp or 16bpp).");

    MAPSUInt32 outputChanSeq = MAPS_CHANNELSEQ_RGB;
    switch (m_outputFormat)
    {
    case OUTPUT_FORMAT::BGR:
        outputChanSeq = MAPS_CHANNELSEQ_BGR;
        break;
    case OUTPUT_FORMAT::RGB:
        outputChanSeq = MAPS_CHANNELSEQ_RGB;
        break;
    case OUTPUT_FORMAT::BGRA:
        outputChanSeq = MAPS_CHANNELSEQ_BGRA;
        break;
    case OUTPUT_FORMAT::RGBA:
        outputChanSeq = MAPS_CHANNELSEQ_RGBA;
        break;
    }

    IplImage model = MAPS::IplImageModel(imageIn.width, imageIn.height, outputChanSeq, imageIn.dataOrder, imageIn.depth, imageIn.align);

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

void MAPSBayerDecoder::AllocateOutputBufferMaps(const MAPSTimestamp, const MAPS::InputElt<MAPSImage> imageInElt)
{
    const MAPSImage& imageIn = imageInElt.Data();

    MAPSUInt32 outputChanSeq;
    switch (m_outputFormat)
    {
    case OUTPUT_FORMAT::BGR:
        outputChanSeq = MAPS_CHANNELSEQ_BGR;
        break;
    case OUTPUT_FORMAT::RGB:
        outputChanSeq = MAPS_CHANNELSEQ_RGB;
        break;
    case OUTPUT_FORMAT::BGRA:
        outputChanSeq = MAPS_CHANNELSEQ_BGRA;
        break;
    case OUTPUT_FORMAT::RGBA:
        outputChanSeq = MAPS_CHANNELSEQ_RGBA;
        break;
    }

    MAPSUInt32 fourcc = 0;
    MAPS::Memcpy((char*)&fourcc, (const char*)imageIn.imageCoding, 4);
    MAPSInt32 depth = IPL_DEPTH_8U;
    switch (fourcc)
    {
    case MAPS_IMAGECODING_RGGB:
    case MAPS_IMAGECODING_GRBG:
    case MAPS_IMAGECODING_GBRG:
    case MAPS_IMAGECODING_BA81:
        depth = IPL_DEPTH_8U;
        break;
    case MAPS_IMAGECODING_RG10:
    case MAPS_IMAGECODING_BA10:
    case MAPS_IMAGECODING_GB10:
    case MAPS_IMAGECODING_BG10:
    case MAPS_IMAGECODING_RG12:
    case MAPS_IMAGECODING_BA12:
    case MAPS_IMAGECODING_GB12:
    case MAPS_IMAGECODING_BG12:
    case MAPS_IMAGECODING_RG16:
    case MAPS_IMAGECODING_GR16:
    case MAPS_IMAGECODING_GB16:
    case MAPS_IMAGECODING_BYR2:
    {
        depth = IPL_DEPTH_16U;
    }
    break;
    default:
        Error("Image coding not supported");
    }

    // Create a new IplImage to allocate the output buffer using the channel sequence determined above
    IplImage model = MAPS::IplImageModel(imageIn.width, imageIn.height, outputChanSeq, IPL_DATA_ORDER_PIXEL, depth, IPL_ALIGN_QWORD);

    if (m_gpuMatAsOutput)
    {
        try
        {
            AllocateDynamicOutputBuffers(
                DynamicOutput<MapsCudaStruct>(Output("o_gpu"),
                    [&] { return new MapsCudaStruct(imageIn.imageSize, model); }  // struct allocation
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

void MAPSBayerDecoder::AllocateOutputBufferGpu(const MAPSTimestamp, const MAPS::InputElt<MapsCudaStruct> imageInElt)
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
        const IplImage& proxy = imageIn.m_IplImageProxy;
        IplImage model = MAPS::IplImageModel(proxy.width, proxy.height, proxy.channelSeq, proxy.dataOrder, proxy.depth, proxy.align);
        Output(0).AllocOutputBufferIplImage(model);
    }
}

void MAPSBayerDecoder::ProcessDataIpl(const MAPSTimestamp ts, const MAPS::InputElt<IplImage> inElt)
{
    MAPS::OutputGuard<> outGuard{ this, Output(0) };
    m_tempImageIn = convTools::noCopyIplImage2Mat(&inElt.Data());

    if (m_useCuda)
    {
        cv::cuda::GpuMat src(m_tempImageIn);

        if (m_gpuMatAsOutput)
        {
            MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
            const IplImage& proxy = outputData.m_IplImageProxy;
            cv::cuda::GpuMat dst(proxy.height, proxy.width, CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels), outputData.m_points);

            try {
                // Convert an image from one color space to another depending on the pattern use
                switch (m_pattern)
                {
                case MAPS_BAYER_PATTERN_BG:
                    cv::cuda::cvtColor(src, dst, m_colorConvCode);
                    break;
                case MAPS_BAYER_PATTERN_GB:
                    cv::cuda::cvtColor(src, dst, m_colorConvCode);
                    break;
                case MAPS_BAYER_PATTERN_RG:
                    cv::cuda::cvtColor(src, dst, m_colorConvCode);
                    break;
                case MAPS_BAYER_PATTERN_GR:
                    cv::cuda::cvtColor(src, dst, m_colorConvCode);
                    break;
                }
            }
            catch (const std::exception& e)
            {
                Error(e.what());
            }
        }
        else
        {
            cv::cuda::GpuMat dst;
            IplImage& imageOut = outGuard.DataAs<IplImage>();
            m_tempImageOut = convTools::noCopyIplImage2Mat(&imageOut);

            try {
                // Convert an image from one color space to another depending on the pattern use
                switch (m_pattern)
                {
                case MAPS_BAYER_PATTERN_BG:
                    cv::cuda::cvtColor(src, dst, m_colorConvCode);
                    break;
                case MAPS_BAYER_PATTERN_GB:
                    cv::cuda::cvtColor(src, dst, m_colorConvCode);
                    break;
                case MAPS_BAYER_PATTERN_RG:
                    cv::cuda::cvtColor(src, dst, m_colorConvCode);
                    break;
                case MAPS_BAYER_PATTERN_GR:
                    cv::cuda::cvtColor(src, dst, m_colorConvCode);
                    break;
                }
            }
            catch (const std::exception& e)
            {
                Error(e.what());
            }
            dst.download(m_tempImageOut);
        }
    }
    else
    {
        IplImage& imageOut = outGuard.DataAs<IplImage>();
        m_tempImageOut = convTools::noCopyIplImage2Mat(&imageOut); // Convert IplImage to cv::Mat without copying

        try {
            // Convert an image from one color space to another depending on the pattern use
            switch (m_pattern)
            {
            case MAPS_BAYER_PATTERN_BG:
                cv::cvtColor(m_tempImageIn, m_tempImageOut, m_colorConvCode);
                break;
            case MAPS_BAYER_PATTERN_GB:
                cv::cvtColor(m_tempImageIn, m_tempImageOut, m_colorConvCode);
                break;
            case MAPS_BAYER_PATTERN_RG:
                cv::cvtColor(m_tempImageIn, m_tempImageOut, m_colorConvCode);
                break;
            case MAPS_BAYER_PATTERN_GR:
                cv::cvtColor(m_tempImageIn, m_tempImageOut, m_colorConvCode);
                break;
            }
        }
        catch (const std::exception& e)
        {
            Error(e.what());
        }

        if (static_cast<void*>(m_tempImageOut.data) != static_cast<void*>(imageOut.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
            Error("cv::Mat data ptr and imageOut data ptr are different.");
    }

    outGuard.VectorSize() = 0;
    outGuard.Timestamp() = ts;
}

void MAPSBayerDecoder::ProcessDataMaps(const MAPSTimestamp ts, const MAPS::InputElt<MAPSImage> inElt)
{
    MAPS::OutputGuard<> outGuard{ this, Output(0) };

    const MAPSImage& imageIn = inElt.Data();

    MAPSUInt32 fourcc = 0;
    MAPS::Memcpy((char*)&fourcc, (const char*)imageIn.imageCoding, 4);
    switch (fourcc)
    {
    case MAPS_IMAGECODING_RGGB:
    case MAPS_IMAGECODING_GRBG:
    case MAPS_IMAGECODING_GBRG:
    case MAPS_IMAGECODING_BA81:
    {
        m_tempImageIn = cv::Mat(imageIn.height, imageIn.width, CV_8UC1, imageIn.imageData);
    }
    break;
    case MAPS_IMAGECODING_RG10:
    case MAPS_IMAGECODING_BA10:
    case MAPS_IMAGECODING_GB10:
    case MAPS_IMAGECODING_BG10:
    case MAPS_IMAGECODING_RG12:
    case MAPS_IMAGECODING_BA12:
    case MAPS_IMAGECODING_GB12:
    case MAPS_IMAGECODING_BG12:
    case MAPS_IMAGECODING_RG16:
    case MAPS_IMAGECODING_GR16:
    case MAPS_IMAGECODING_GB16:
    case MAPS_IMAGECODING_BYR2:
    {
        m_tempImageIn = cv::Mat(imageIn.height, imageIn.width, CV_16UC1, imageIn.imageData);
    }
    break;
    default:
        Error("Image coding not supported");
    }

    if (m_useCuda)
    {
        cv::cuda::GpuMat src(m_tempImageIn);

        if (m_gpuMatAsOutput)
        {
            MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
            const IplImage& proxy = outputData.m_IplImageProxy;
            cv::cuda::GpuMat dst(proxy.height, proxy.width, CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels), outputData.m_points);
            ConvertGpu(src, dst);
        }
        else
        {
            cv::cuda::GpuMat dst;
            IplImage& imageOut = outGuard.DataAs<IplImage>();
            m_tempImageOut = convTools::noCopyIplImage2Mat(&imageOut);
            ConvertGpu(src, dst);
            dst.download(m_tempImageOut);

            if (static_cast<void*>(m_tempImageOut.data) != static_cast<void*>(imageOut.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
                Error("cv::Mat data ptr and imageOut data ptr are different.");
        }
    }
    else
    {
        IplImage& imageOut = outGuard.DataAs<IplImage>();
        m_tempImageOut = convTools::noCopyIplImage2Mat(&imageOut); // Convert IplImage to cv::Mat without copying

        try {
            // Convert an image from one color space to another depending on the pattern use
            switch (m_pattern)
            {
            case MAPS_BAYER_PATTERN_BG:
                cv::cvtColor(m_tempImageIn, m_tempImageOut, m_colorConvCode);
                break;
            case MAPS_BAYER_PATTERN_GB:
                cv::cvtColor(m_tempImageIn, m_tempImageOut, m_colorConvCode);
                break;
            case MAPS_BAYER_PATTERN_RG:
                cv::cvtColor(m_tempImageIn, m_tempImageOut, m_colorConvCode);
                break;
            case MAPS_BAYER_PATTERN_GR:
                cv::cvtColor(m_tempImageIn, m_tempImageOut, m_colorConvCode);
                break;
            }
        }
        catch (const std::exception& e)
        {
            Error(e.what());
        }

        if (static_cast<void*>(m_tempImageOut.data) != static_cast<void*>(imageOut.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
            Error("cv::Mat data ptr and imageOut data ptr are different.");
    }

    outGuard.VectorSize() = 0;
    outGuard.Timestamp() = ts;
}

void MAPSBayerDecoder::ProcessDataGpu(const MAPSTimestamp ts, const MAPS::InputElt<MapsCudaStruct> inElt)
{
    MAPS::OutputGuard<> outGuard{ this, Output(0) };

    const IplImage& proxy = inElt.Data().m_IplImageProxy;
    const cv::cuda::GpuMat src(proxy.height, proxy.width, CV_MAKETYPE(proxy.depth == 8 ? 0 : proxy.depth / 8, proxy.nChannels), inElt.Data().m_points);

    if (m_gpuMatAsOutput)
    {
        MapsCudaStruct& outputData = outGuard.DataAs<MapsCudaStruct>();
        const IplImage& proxyDst = outputData.m_IplImageProxy;
        cv::cuda::GpuMat dst(proxyDst.height, proxyDst.width, CV_MAKETYPE(proxyDst.depth == 8 ? 0 : proxyDst.depth / 8, proxyDst.nChannels), outputData.m_points);
        ConvertGpu(src, dst);
    }
    else
    {
        IplImage& imageOut = outGuard.DataAs<IplImage>();
        m_tempImageOut = convTools::noCopyIplImage2Mat(&imageOut); // Convert IplImage to cv::Mat without copying
        cv::cuda::GpuMat dst;
        ConvertGpu(src, dst);
        dst.download(m_tempImageOut);

        if (static_cast<void*>(m_tempImageOut.data) != static_cast<void*>(imageOut.imageData)) // if the ptr are different then opencv reallocated memory for the cv::Mat
            Error("cv::Mat data ptr and imageOut data ptr are different.");
    }

    outGuard.VectorSize() = 0;
    outGuard.Timestamp() = ts;
}

void MAPSBayerDecoder::ConvertGpu(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
{
    try {
        // Convert an image from one color space to another depending on the pattern use
        switch (m_pattern)
        {
        case MAPS_BAYER_PATTERN_BG:
            cv::cuda::cvtColor(src, dst, m_colorConvCode);
            break;
        case MAPS_BAYER_PATTERN_GB:
            cv::cuda::cvtColor(src, dst, m_colorConvCode);
            break;
        case MAPS_BAYER_PATTERN_RG:
            cv::cuda::cvtColor(src, dst, m_colorConvCode);
            break;
        case MAPS_BAYER_PATTERN_GR:
            cv::cuda::cvtColor(src, dst, m_colorConvCode);
            break;
        }
    }
    catch (const std::exception& e)
    {
        Error(e.what());
    }
}
