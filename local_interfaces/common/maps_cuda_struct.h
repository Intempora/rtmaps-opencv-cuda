/////////////////////////////////////////////////////////////////////////////////
//
//   Copyright 2018-2024 Intempora S.A.S.
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

#pragma once

#include <cstdint>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <maps.h> 
#define maps_report_callback MAPS::ReportInfo

#pragma pack(push,1)

struct MapsCudaStruct
{
    int m_size; //size in bytes
    IplImage m_IplImageProxy;
    void* m_points;

    MapsCudaStruct(const int width_, const int height_, const int nbChannels_, const IplImage& image)
        : m_size(width_ * height_ * nbChannels_)
    {
        m_points = allocateMemory(m_size);

        m_IplImageProxy.align = image.align;
        m_IplImageProxy.depth = image.depth;
        m_IplImageProxy.dataOrder = image.dataOrder;
        m_IplImageProxy.nChannels = image.nChannels;
        m_IplImageProxy.width = image.width;
        m_IplImageProxy.height = image.height;
        memcpy(m_IplImageProxy.channelSeq, image.channelSeq, 4);

        //std::ostringstream oss;
        //oss << "New " << toString();
        //maps_report_callback(oss.str().c_str());
    }

    MapsCudaStruct(const int size_, const IplImage& image)
        : m_size(size_)
    {
        m_points = allocateMemory(m_size);

        m_IplImageProxy.align = image.align;
        m_IplImageProxy.depth = image.depth;
        m_IplImageProxy.dataOrder = image.dataOrder;
        m_IplImageProxy.nChannels = image.nChannels;
        m_IplImageProxy.width = image.width;
        m_IplImageProxy.height = image.height;
        memcpy(m_IplImageProxy.channelSeq, image.channelSeq, 4);

        //std::ostringstream oss;
        //oss << "New " << toString();
        //maps_report_callback(oss.str().c_str());
    }

    MapsCudaStruct(const MapsCudaStruct& cudaStruct)
        : m_size(cudaStruct.m_size)
    {
        m_points = allocateMemory(m_size);

        m_IplImageProxy.align = cudaStruct.m_IplImageProxy.align;
        m_IplImageProxy.depth = cudaStruct.m_IplImageProxy.depth;
        m_IplImageProxy.dataOrder = cudaStruct.m_IplImageProxy.dataOrder;
        m_IplImageProxy.nChannels = cudaStruct.m_IplImageProxy.nChannels;
        m_IplImageProxy.width = cudaStruct.m_IplImageProxy.width;
        m_IplImageProxy.height = cudaStruct.m_IplImageProxy.height;
        memcpy(m_IplImageProxy.channelSeq, cudaStruct.m_IplImageProxy.channelSeq, 4);

        //std::ostringstream oss;
        //oss << "New " << toString();
        //maps_report_callback(oss.str().c_str());
    }

    ~MapsCudaStruct()
    {
        //std::ostringstream oss;
        //oss << "Delete " << toString();
        //maps_report_callback(oss.str().c_str());

        freeMemory(m_points);
    }

    static void* allocateMemory(const int nbPoints_)
    {
        void* points = nullptr;
        const cudaError_t cudaErr = cudaMalloc(&points, nbPoints_);

        if (cudaErr != cudaError_t::cudaSuccess)
        {
            maps_report_callback("Allocation failed");
            throw std::runtime_error("Allocation failed");
        }

        return points;
    }

    static void freeMemory(void* points_)
    {
        if (points_ != nullptr)
        {
            cudaFree(points_);
        }
    }

    std::string toString() const
    {
        std::ostringstream oss;
        oss << "MyCudaStruct [this:" << this << "] (Size:" << m_size << ")";
        return oss.str();
    }
};

#pragma pack(pop)

#ifdef MAPS_FILTER_USER_DYNAMIC_STRUCTURE
const MAPSTypeFilterBase Filter_MapsCudaStruct = MAPS_FILTER_USER_DYNAMIC_STRUCTURE(MapsCudaStruct);
#endif