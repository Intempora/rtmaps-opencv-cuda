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
#include "maps/input_reader/maps_input_reader.hpp"
#include "maps_OpenCV_Conversion.h"

#include "common/maps_dynamic_custom_struct_component.h"
#include "common/maps_cuda_struct.h"

namespace
{
    const int CS_RGB24 = 0;
    const int CS_BGR24 = 1;
    const int CS_YUV24 = 2;
    const int CS_HSV = 3;
    const int CS_GRAY = 4;
    const int CS_RGBA = 5;
    const int CS_BGRA = 6;
    const int CS_AUTO = 7;
}

// Declares a new MAPSComponent child class
class MAPSColorSpaceConverter : public MAPS_DynamicCustomStructComponent
{
    // Use standard header definition macro
    MAPS_CHILD_COMPONENT_HEADER_CODE(MAPSColorSpaceConverter, MAPS_DynamicCustomStructComponent)

    void Set(MAPSProperty& p, const MAPSString& value) override;
    void Set(MAPSProperty& p, const MAPSEnumStruct& enum_prop) override;
    void Dynamic() override;
    void FreeBuffers() override;

private:
    void AllocateOutputBufferSize(const MAPSTimestamp /*ts*/, const MAPS::InputElt<IplImage> imageInElt);
    void ProcessData(const MAPSTimestamp ts, const MAPS::InputElt<IplImage> inElt);
    void AllocateOutputBufferSizeGpu(const MAPSTimestamp /*ts*/, const MAPS::InputElt<MapsCudaStruct> imageInElt);
    void ProcessDataGpu(const MAPSTimestamp ts, const MAPS::InputElt<MapsCudaStruct> inElt);
    void CheckInputColorSpace(int chanSeq);
    void ConvertGpu(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);

private :
    // Place here your specific methods and attributes
    int m_inputCS;
    int m_outputCS;
    int m_openCVConvertCode;
    bool m_useCuda;
    bool m_gpuMatAsInput = false;
    bool m_gpuMatAsOutput = false;

    std::array<cv::Mat, 3> m_tempChannels;
    cv::Mat m_workImage;


    std::unique_ptr<MAPS::InputReader> m_inputReader;
};
