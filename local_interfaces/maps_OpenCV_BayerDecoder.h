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

#pragma once

// Includes maps sdk library header
#include "maps_OpenCV_Conversion.h"
#include "maps/input_reader/maps_input_reader.hpp"
#include "common/maps_dynamic_custom_struct_component.h"
#include "common/maps_cuda_struct.h"

enum OUTPUT_FORMAT : uint8_t
{
    BGR,
    RGB,
    BGRA,
    RGBA
};

// Declares a new MAPSComponent child class
class MAPSBayerDecoder : public MAPS_DynamicCustomStructComponent
{
    // Use standard header definition macro
    MAPS_CHILD_COMPONENT_HEADER_CODE(MAPSBayerDecoder, MAPS_DynamicCustomStructComponent)

    void Set(MAPSProperty& p, const MAPSString& value) override;
    void Dynamic() override;
    void FreeBuffers() override;

private:
    void AllocateOutputBufferIpl(const MAPSTimestamp /*ts*/, const MAPS::InputElt<IplImage> imageInElt);
    void AllocateOutputBufferMaps(const MAPSTimestamp /*ts*/, const MAPS::InputElt<MAPSImage> imageInElt);
    void AllocateOutputBufferGpu(const MAPSTimestamp /*ts*/, const MAPS::InputElt<MapsCudaStruct> imageInElt);
    void ProcessDataIpl(const MAPSTimestamp ts, const MAPS::InputElt<IplImage> inElt);
    void ProcessDataMaps(const MAPSTimestamp ts, const MAPS::InputElt<MAPSImage> inElt);
    void ProcessDataGpu(const MAPSTimestamp ts, const MAPS::InputElt<MapsCudaStruct> inElt);

    void ConvertGpu(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);

private :
    // Place here your specific methods and attributes
    OUTPUT_FORMAT m_outputFormat;
    cv::ColorConversionCodes m_colorConvCode;
    int	 m_pattern;
    bool m_useCuda;
    bool m_gpuMatAsInput = false;
    bool m_gpuMatAsOutput = false;

    cv::Mat m_tempImageIn;
    cv::Mat m_tempImageOut;

    std::unique_ptr<MAPS::InputReader> m_inputReader;
};
