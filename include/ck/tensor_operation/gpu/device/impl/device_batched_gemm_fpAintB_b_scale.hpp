// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_gemm_xdl_cshuffle_v3_b_scale.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename BScaleDataType,
          typename CDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t ScaleBlockN, // scale block for N
          index_t ScaleBlockK, // scale block for K
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          bool PermuteA                               = false,
          bool PermuteB                               = false>
struct DeviceBatchedGemm_Xdl_CShuffleV3 : public DeviceBatchedGemm_BScale<ALayout,
                                                                          BLayout,
                                                                          CLayout,
                                                                          ADataType,
                                                                          BDataType,
                                                                          BScaleDataType,
                                                                          CDataType,
                                                                          ScaleBlockN,
                                                                          ScaleBlockK,
                                                                          AElementwiseOperation,
                                                                          BElementwiseOperation,
                                                                          CElementwiseOperation>
{
    using GridwiseGemm = GridwiseGemm_xdl_cshuffle_v3<
        ALayout,
        BLayout,
        CLayout,
        ADataType,
        BDataType,
        AccDataType,
        CShuffleDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        BlockSize,
        ScaleBlockN,
        ScaleBlockK,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CBlockTransferScalarPerVector_NPerBlock,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        PermuteA,
        PermuteB>;

    // Argument
    using Argument = typename GridwiseGemm::Argument;

    // Invoker
    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            if(!GridwiseGemm::CheckValidity(arg))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            // const index_t grid_size =
            //    GridwiseGemm::CalculateGridSize(arg.e_grid_desc_m_n_) * arg.Batch_;

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(arg.M, arg.N, 1);

            const index_t grid_size = gdx * gdy * gdz * arg.Batch;

            printf("gdx: %d, gdy: %d, gdz: %d, grid_size: %d, batch: %d\n",gdx, gdy, gdz, grid_size, arg.Batch);

            index_t k_split = (arg.K + KPerBlock - 1) / KPerBlock * KPerBlock;
            
            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(k_split);

            float ave_time = 0;

            const auto Run =
                [&](const auto& kernel) {
                    ave_time = launch_and_time_kernel(
                        stream_config, kernel, dim3(grid_size), dim3(BlockSize), 0, arg);
                };

            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave
                    ? (BlkGemmPipelineVer == BlockGemmPipelineVersion::v3 &&
                       MPerBlock * NPerBlock * KPerBlock * sizeof(ADataType) <= 128 * 128 * 64 * 2)
                          ? 2
                          : 1
                    : 2;
            
            if(has_main_k_block_loop)
            {
                const auto kernel =
                    kernel_batched_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                        true,
                                                        InMemoryDataOperationEnum::Set,
                                                        minimum_occupancy>;
                Run(kernel);
            }
            else
            {
                const auto kernel =
                    kernel_batched_gemm_xdl_cshuffle_v3<GridwiseGemm,
                                                        false,
                                                        InMemoryDataOperationEnum::Set,
                                                        minimum_occupancy>;
                Run(kernel);
            }
        
            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_xdl_supported())
        {
            return false;
        }
        
        // if(!is_bf16_atomic_supported() && std::is_same_v<CDataType, ck::bhalf_t> && arg.KBatch > 1)
        // {
        //     return false;
        // }

        if((arg.K % AK1 != 0 || arg.K % BK1 != 0) && !(GemmSpec == GemmSpecialization::MKPadding ||
                                                       GemmSpec == GemmSpecialization::NKPadding ||
                                                       GemmSpec == GemmSpecialization::MNKPadding ||
                                                       GemmSpec == GemmSpecialization::KPadding))
        {
            return false;
        }

        return GridwiseGemm::CheckValidity(arg);
    }

    index_t GetKPerBlock() override { return KPerBlock; }

    bool GetPermuteB() override { return PermuteB; }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             index_t StrideScaleB,
                             index_t BatchStrideA,
                             index_t BatchStrideB,
                             index_t BatchStrideC,
                             index_t BatchStrideScaleB,
                             const BScaleDataType* p_b_scale,
                             index_t Batch,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        StrideScaleB,
                        BatchStrideA,
                        BatchStrideB,
                        BatchStrideC,
                        BatchStrideScaleB,
                        p_b_scale,
                        Batch,
                        a_element_op,
                        b_element_op,
                        c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        void* p_c,
                        index_t M,
                        index_t N,
                        index_t K,
                        index_t StrideA,
                        index_t StrideB,
                        index_t StrideC,
                        index_t StrideScaleB,
                        index_t BatchStrideA,
                        index_t BatchStrideB,
                        index_t BatchStrideC,
                        index_t BatchStrideScaleB,
                        const void* p_b_scale,
                        index_t Batch,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          StrideScaleB,
                                          BatchStrideA,
                                          BatchStrideB,
                                          BatchStrideC,
                                          BatchStrideScaleB,
                                          static_cast<const BScaleDataType*>(p_b_scale),
                                          Batch,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceBatchedGemm_Xdl_CShuffleV3"
            << "<"
            <<std::string(ALayout::name)[0]
            <<std::string(BLayout::name)[0]
            <<std::string(CLayout::name)[0]
            <<">"
            << " BlkSize: "
            << BlockSize << ", "
            << "BlkTile: "
            << MPerBlock<<"x"<<NPerBlock<<"x"<<KPerBlock << ", "
            << "WaveTile: "
            << MPerXDL<<"x"<<NPerXDL << ", "
            << "WaveMap: "
            << MXdlPerWave<<"x" << NXdlPerWave<<", "
            << "VmemReadVec: "
            << ABlockTransferSrcScalarPerVector<<"x"<<BBlockTransferSrcScalarPerVector<<", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseGemm::BlockwiseGemmPipe::PrefetchStages
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
