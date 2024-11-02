// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_b_scale.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using I4  = pk_i4_t;
using F16 = half_t;
using F32 = float;

using Row = tensor_layout::gemm::RowMajor;
using Col = tensor_layout::gemm::ColumnMajor;

template <index_t... Is>
using S = Sequence<Is...>;

using PassThrough = element_wise::PassThrough;

static constexpr auto GemmDefault    = GemmSpecialization::Default;
static constexpr auto GemmKPadding   = GemmSpecialization::KPadding;
static constexpr auto GemmMNPadding  = GemmSpecialization::MNPadding;
static constexpr auto GemmMNKPadding = GemmSpecialization::MNKPadding;

static constexpr auto Intrawave = BlockGemmPipelineScheduler::Intrawave;
static constexpr auto Interwave = BlockGemmPipelineScheduler::Interwave;

#if 0
template <GemmSpecialization GemmSpec>
using device_gemm_xdl_b_scale_f16_i4_f16_mk_nk_mn_comp_instances = std::tuple<

#endif

template <BlockGemmPipelineScheduler BlkGemmPipeSched, GemmSpecialization GemmSpec>
using device_gemm_b_scale_xdl_f16_i4_f16_mk_nk_mn_mem_instances = std::tuple<
    // clang-format off
        //#########################| ALayout| BLayout| CLayout|AData| BData| BScale| CData| AccData| Cshuffle|           A|           B|           C|          GEMM| Block| Scale| Scale|  MPer|  NPer|  KPer| AK1| BK1|MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|    Block-wiseGemm|               Block-wiseGemm|
        //#########################|        |        |        | Type|  Type|   Data|  Type|    Type|     Type| Elementwise| Elementwise| Elementwise|Specialization|  Size| Block| Block| Block| Block| Block|    |    | XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|          Pipeline|                     Pipeline|
        //#########################|        |        |        |     |      |   Type|      |        |         |   Operation|   Operation|   Operation|              |      |     N|     K|      |      |      |    |    |Wave| Wave|     |     | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|         Scheduler|                     Verision|
        //#########################|        |        |        |     |      |       |      |        |         |            |            |            |              |      |      |      |      |      |      |    |    |    |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |                  |                             |
        
        //Compute friendly
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    1,   128,   128,  128,   128,   8,   32,  32,   32,    2,    2,     S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    1,   128,   128,  128,    64,   8,   32,  32,   32,    2,    2,     S<8, 32, 1>,      S<1, 0, 2>,    S<1, 0, 2>,             2,              8,              8,          0,    S<2, 128, 1>,    S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v4, half_t, half_t, false, false>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    1,   128,   224,  256,   128,   8,   32,  16,   16,    7,    8,     S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           2,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, true>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    1,   128,   128,  128,   128,   8,   32,  32,   32,    2,    2,     S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
 
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    1,   128,   128,  128,    64,   8,   32,  32,   32,    2,    2,     S<8, 32, 1>,      S<1, 0, 2>,    S<1, 0, 2>,             2,              8,              8,          0,    S<2, 128, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v4, half_t, half_t, false, false>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    1,   128,   224,  256,   128,   8,   32,  16,   16,    7,    8,     S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           2,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, true>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    1,   128,   128,  128,    64,   8,   32,  32,   32,    2,    2,     S<8, 32, 1>,      S<1, 0, 2>,    S<1, 0, 2>,             2,              8,              8,          0,    S<2, 128, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        
        //Latency friendly
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    32,   16,   128,   8,   16,  16,   16,    1,    1,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               2,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,    64,    1,   128,    16,   16,   128,   8,   16,  16,   16,    1,    1,     S<16, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 8, 1>,      S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 4>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,    64,    1,   128,    16,   16,   128,   8,   16,  16,   16,    1,    1,     S<8,  8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 8, 1>,      S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 4>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    16,   32,   128,   8,   32,  16,   16,    1,    1,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        
        // Memory friendly v3
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,   128,   32,   128,   8,   32,  32,   32,    2,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,   128,   16,   128,   8,   16,  16,   16,    4,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               2,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    64,   32,   128,   8,   32,  32,   32,    1,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    64,   16,   128,   8,   16,  16,   16,    2,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               2,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    32,   16,   128,   8,   16,  16,   16,    1,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               2,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,    64,    1,   128,    16,   16,   128,   8,   16,  16,   16,    1,    1,     S<16, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 8, 1>,      S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 4>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,    64,    1,   128,    16,   16,   128,   8,   16,  16,   16,    1,    1,     S<16, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 8, 1>,      S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 4>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    16,   32,   128,   8,   32,  16,   16,    1,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    16,   64,   128,   8,   32,  16,   16,    1,    2,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    32,   64,   128,   8,   32,  32,   32,    1,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    16,  128,   128,   8,   32,  16,   16,    1,    4,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    32,  128,   128,   8,   32,  32,   32,    1,    2,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    1,   128,    16,  256,   128,   8,   32,  16,   16,    1,    4,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 16>,              4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    1,   128,    32,  256,   128,   8,   32,  32,   32,    1,    2,     S<16, 16, 1>,    S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 16>,              8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,

        // Memory friendly v4
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,   128,   32,   64,    8,   32,  32,   32,    2,    1,     S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<2, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v4, half_t, half_t, false, true>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,   128,   16,   128,   8,   16,  16,   16,    4,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               2,  BlkGemmPipeSched, BlockGemmPipelineVersion::v4, half_t, half_t, false, true>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    64,   32,   128,   8,   32,  32,   32,    1,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v4, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    64,   16,   128,   8,   16,  16,   16,    2,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               2,  BlkGemmPipeSched, BlockGemmPipelineVersion::v4, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    32,   16,   128,   8,   16,  16,   16,    1,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               2,  BlkGemmPipeSched, BlockGemmPipelineVersion::v4, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,    64,    1,   128,    16,   16,   128,   8,   16,  16,   16,    1,    1,     S<16, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 8, 1>,      S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 4>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v4, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,    64,    1,   128,    16,   16,   128,   8,   16,  16,   16,    1,    1,     S<16, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 8, 1>,      S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 4>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v4, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    16,   32,   128,   8,   32,  16,   16,    1,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v4, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    16,   64,   128,   8,   32,  16,   16,    1,    2,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v4, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    32,   64,   128,   8,   32,  32,   32,    1,    1,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v4, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    16,  128,   128,   8,   32,  16,   16,    1,    4,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v4, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    32,  128,   128,   8,   32,  32,   32,    1,    2,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v4, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    1,   128,    16,  256,   128,   8,   32,  16,   16,    1,    4,     S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 16>,              4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v4, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    1,   128,    32,  256,   128,   8,   32,  32,   32,    1,    2,     S<16, 16, 1>,    S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 16>,              8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v4, half_t, half_t, false, false>,

        //new Compute friendly kernel
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    1,   128,   128,   128,   64,   8,   32,  32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<2, 128, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    1,   128,   128,   128,   64,   8,   32,  32,   32,    4,    1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<2, 128, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 32, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,

        //new Memory friendly kernel
        DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    1,   128,   16,    64,   256,   8,   32,  16,   16,    1,    1,     S<32, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 32, 1>,      S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>
        
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    32,   16,   256,   8,   16,  16,   16,    1,    1,     S<32, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<16, 8, 1>,      S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               2,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,    64,    1,   128,    16,   16,   256,   8,   16,  16,   16,    1,    1,     S<32, 2, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<16, 4, 1>,      S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 4>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,    64,    1,   128,    16,   16,   256,   8,   16,  16,   16,    1,    1,     S<32, 2, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<16, 4, 1>,      S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 4>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    16,   32,   256,   8,   32,  16,   16,    1,    1,     S<32, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 16, 1>,      S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        
        // // Memory friendly v3
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    64,   32,   256,   8,   32,  32,   32,    1,    1,     S<32, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    64,   16,   256,   8,   16,  16,   16,    2,    1,     S<32, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               2,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    32,   16,   256,   8,   16,  16,   16,    1,    1,     S<32, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<16, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 8>,               2,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,    64,    1,   128,    16,   16,   256,   8,   16,  16,   16,    1,    1,     S<32, 2, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<16, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 4>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,    64,    1,   128,    16,   16,   256,   8,   16,  16,   16,    1,    1,     S<32, 2, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<16, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 16, 1, 4>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    16,   32,   256,   8,   32,  16,   16,    1,    1,     S<32, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    16,   64,   256,   8,   32,  16,   16,    1,    2,     S<32, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    32,   64,   256,   8,   32,  32,   32,    1,    1,     S<32, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    16,  128,   256,   8,   32,  16,   16,    1,    4,     S<32, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   128,    1,   128,    32,  128,   256,   8,   32,  32,   32,    1,    2,     S<32, 4, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 8>,               8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    1,   128,    16,  256,   256,   8,   32,  16,   16,    1,    4,     S<32, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 16>,              4,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>,
        // DeviceGemm_Xdl_CShuffleV3<  Row,     Col,     Row,     F16,    I4,   F16,   F16,   F32,     F16,      PassThrough, PassThrough, PassThrough,       GemmSpec,   256,    1,   128,    32,  256,   256,   8,   32,  32,   32,    1,    2,     S<32, 8, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             32,             32,          0,          1,           1,                   S<1, 16, 1, 16>,              8,  BlkGemmPipeSched, BlockGemmPipelineVersion::v3, half_t, half_t, false, false>
    // clang-format on
    >;
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
