// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3r1.hpp"

using ADataType        = ck::bhalf_t;
using BDataType        = ck::bhalf_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;
using CDataType        = ck::bhalf_t;
using D0DataType       = ck::bhalf_t;
using DsDataType       = ck::Tuple<D0DataType>;

using ALayout  = Row;
using BLayout  = Col;
using CLayout  = Row;
using D0Layout = CLayout;
using DsLayout = ck::Tuple<D0Layout>;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CElementOp   = PassThrough;
using CDEElementOp = Add;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNPadding;

// clang-format off
using DeviceGemmV2Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3R1<
        ALayout,   BLayout,  CLayout,   DsLayout,
        ADataType,   BDataType,  CDataType, DsDataType, AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        256,
        128, 128, 
        64, 8, 8,
        16,   16,
        4,    4,
        S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        1, 2, S<1, 32, 1, 8>, 8,
        ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

// using ReferenceGemmInstance =
//             ck::tensor_operation::host::ReferenceGemmMultipleD<ADataType,
//                                                                BDataType,
//                                                                ck::Tuple<D0DataType>,
//                                                                CDataType,
//                                                                AccDataType,
//                                                                AElementOp,
//                                                                BElementOp,
//                                                                CDEElementOp>;

#include "run_gemm_splitk_reduce_multi_d_example.inc"

int main(int argc, char* argv[]) { return !run_gemm_splitk_example(argc, argv); }
