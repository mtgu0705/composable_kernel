#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_fpAintB_b_scale.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/utility/literals.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType        = F16;
using BDataType        = ck::pk_i4_t;
using BScaleDataType   = ck::half_t;
using AccDataType      = F32;
using CShuffleDataType = F16;
using CDataType        = F16;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

static constexpr bool PermuteB = false;

static constexpr ck::index_t Scale_Block_N = 1;
static constexpr ck::index_t Scale_Block_K = 128;

static constexpr ck::index_t KPerBlock = 64;

// clang-format off
using DeviceBatchedGemmV2Instance = 
    ck::tensor_operation::device::DeviceBatchedGemm_Xdl_CShuffleV3<
        ALayout,   BLayout,  CLayout,   
        ADataType, BDataType, BScaleDataType, CDataType, AccDataType, CShuffleDataType, 
        AElementOp, BElementOp, CElementOp, GemmDefault, 
        256, Scale_Block_N, Scale_Block_K,
        128, 128,
        KPerBlock, 8, 32,
        32,   32,
        2,    2,
        S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 8, 8, 0,
        S<2, 128, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 32, 32, 0,
        1, 1, S<1, 32, 1, 8>, 8,
        
        // 256, Scale_Block_N, Scale_Block_K,
        // 16, 64,
        // KPerBlock, 8, 32,
        // 16,   16,
        // 1,    1,
        // S<32, 8, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        // 2, 8, 8, 0,
        // S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        // 2, 32, 32, 0,
        // 1, 1, S<1, 16, 1, 8>, 8,
        ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, CDataType, CDataType, false, PermuteB>;
// clang-format on

using ReferenceBatchedGemmInstance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                                                      AccDataType,
                                                                                      CDataType,
                                                                                      AccDataType,
                                                                                      AElementOp,
                                                                                      BElementOp,
                                                                                      CElementOp>;

struct ProblemSize final
{
    ck::index_t M = 128;
    ck::index_t N = 128;
    ck::index_t K = 512;

    ck::index_t stride_A = K;
    ck::index_t stride_B = K;
    ck::index_t stride_C = N;

    ck::index_t batch_stride_A = M * K;
    ck::index_t batch_stride_B = K * N;
    ck::index_t batch_stride_C = M * N;

    ck::index_t batch_count = 2;
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = true;
};

template <typename DataType>
inline __host__ __device__ constexpr double get_rtol()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 1e-1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 1.5e-1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

template <typename DataType>
inline __host__ __device__ constexpr double get_atol()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 16.1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 8192.1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

bool run_batched_gemm(const ProblemSize& problem_size, const ExecutionConfig& config)
{
    using namespace ck::literals;

    auto& [M,
           N,
           K,
           stride_A,
           stride_B,
           stride_C,
           batch_stride_A,
           batch_stride_B,
           batch_stride_C,
           batch_count] = problem_size;

    auto f_host_tensor_descriptor = [](std::size_t batch_count_,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       std::size_t batch_stride,
                                       auto layout) {
        if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
        {
            return HostTensorDescriptor({batch_count_, row, col}, {batch_stride, stride, 1_uz});
        }
        else
        {
            return HostTensorDescriptor({batch_count_, row, col}, {batch_stride, 1_uz, stride});
        }
    };

    ck::index_t Scale_Stride_BN = (K + Scale_Block_K - 1) / Scale_Block_K;
    ck::index_t batch_BScale_Stride =
        ((K + Scale_Block_K - 1) / Scale_Block_K) * ((N + Scale_Block_N - 1) / Scale_Block_N);

    Tensor<ADataType> a_g_m_k(
        f_host_tensor_descriptor(batch_count, M, K, stride_A, batch_stride_A, ALayout{}));
    Tensor<BDataType> b_g_k_n(
        f_host_tensor_descriptor(batch_count, K, N, stride_B, batch_stride_B, BLayout{}));
    Tensor<BDataType> b_g_k_n_permute(
        f_host_tensor_descriptor(batch_count, K, N, stride_B, batch_stride_B, BLayout{}));
    Tensor<BScaleDataType> b1_g_k_n(
        f_host_tensor_descriptor(batch_count,
                                 (K + Scale_Block_K - 1) / Scale_Block_K,
                                 (N + Scale_Block_N - 1) / Scale_Block_N,
                                 Scale_Stride_BN,
                                 batch_BScale_Stride,
                                 BLayout{}));

    switch(config.init_method)
    {
    case 0:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        b1_g_k_n.GenerateTensorValue(GeneratorTensor_1<BScaleDataType>{1});
        break;
    case 1:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b1_g_k_n.GenerateTensorValue(GeneratorTensor_3<BScaleDataType>{0, 1.0});
        break;
    case 2:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b1_g_k_n.GenerateTensorValue(GeneratorTensor_1<BScaleDataType>{1});
        break;
    case 3:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        b1_g_k_n.GenerateTensorValue(GeneratorTensor_1<BScaleDataType>{1});
        break;
    case 4:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        b1_g_k_n.GenerateTensorValue(GeneratorTensor_3<BScaleDataType>{0, 1.0});
        break;
    case 5:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b1_g_k_n.GenerateTensorValue(GeneratorTensor_1<BScaleDataType>{1});
        break;
    default:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.5, 0.5});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        b1_g_k_n.GenerateTensorValue(GeneratorTensor_3<BScaleDataType>{0, 1.0});
    }

    Tensor<CDataType> c_g_m_n_host_result(
        f_host_tensor_descriptor(batch_count, M, N, stride_C, batch_stride_C, CLayout{}));
    Tensor<CDataType> c_g_m_n_device_result(
        f_host_tensor_descriptor(batch_count, M, N, stride_C, batch_stride_C, CLayout{}));

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b_g_k_n: " << b_g_k_n.mDesc << std::endl;
    std::cout << "b1_g_k_n: " << b1_g_k_n.mDesc << std::endl;
    std::cout << "c_g_m_n: " << c_g_m_n_host_result.mDesc << std::endl;

    DeviceMem a_g_m_k_device_buf(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_g_k_n_device_buf(sizeof(BDataType) * b_g_k_n_permute.mDesc.GetElementSpaceSize());
    DeviceMem b1_g_scale_device_buf(sizeof(BScaleDataType) * b1_g_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_g_m_n_device_buf(sizeof(CDataType) *
                                 c_g_m_n_device_result.mDesc.GetElementSpaceSize());

    printf("b_g_k_n element space size: %zu, b_g_k_n device size: %lu, BDataType size: %lu\n",
           b_g_k_n_permute.mDesc.GetElementSpaceSize(),
           sizeof(BDataType) * b_g_k_n_permute.mDesc.GetElementSpaceSize(),
           sizeof(BDataType));

    // weight permute
    if constexpr(PermuteB)
    {
        int K1 = KPerBlock;
        int K0 = K / KPerBlock;

        // int K0, N, K1
        for(int bs = 0; bs < batch_count; bs++)
        {
            for(int j = 0; j < K0; j++)
            {
                for(int i = 0; i < N; i++)
                {
                    for(int jj = 0; jj < K1; jj++)
                    {
                        b_g_k_n_permute(bs * batch_stride_B + j * N * K1 + i * K1 + jj) =
                            b_g_k_n(bs * batch_stride_B + i * K + (j * K1 + jj));
                    }
                }
            }
        }
    }
    else
    {
        b_g_k_n_permute = b_g_k_n;
    }

    // vector pk_i4x4 permute
#if 1
    for(int bs = 0; bs < batch_count; bs++)
    {
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < K; j += 8)
            {
                int input[8];

                for(int k = 0; k < 4; k++)
                {
                    int i4x2         = b_g_k_n_permute(bs, j + k * 2, i);
                    input[k * 2 + 0] = (i4x2 >> 4) & 0xf;
                    input[k * 2 + 1] = (i4x2 >> 0) & 0xf;
                }

                // permute 01234567->20643175
                {
                    int hi   = input[2];
                    int lo   = input[0];
                    int i4x2 = (hi << 4) | lo;

                    b_g_k_n_permute(bs, j + 0, i) = i4x2;
                }

                {
                    int hi   = input[6];
                    int lo   = input[4];
                    int i4x2 = (hi << 4) | lo;

                    b_g_k_n_permute(bs, j + 2, i) = i4x2;
                }

                {
                    int hi   = input[3];
                    int lo   = input[1];
                    int i4x2 = (hi << 4) | lo;

                    b_g_k_n_permute(bs, j + 4, i) = i4x2;
                }

                {
                    int hi   = input[7];
                    int lo   = input[5];
                    int i4x2 = (hi << 4) | lo;

                    b_g_k_n_permute(bs, j + 6, i) = i4x2;
                }
            }
        }
    }
#endif

    a_g_m_k_device_buf.ToDevice(a_g_m_k.mData.data());
    b_g_k_n_device_buf.ToDevice(b_g_k_n_permute.mData.data());
    b1_g_scale_device_buf.ToDevice(b1_g_k_n.mData.data());
    DeviceMem workspace;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // do GEMM
    auto gemm      = DeviceBatchedGemmV2Instance{};
    auto invoker   = gemm.MakeInvoker();
    float ave_time = 0;

    auto argument =
        gemm.MakeArgument(static_cast<ADataType*>(a_g_m_k_device_buf.GetDeviceBuffer()),
                          static_cast<BDataType*>(b_g_k_n_device_buf.GetDeviceBuffer()),
                          static_cast<CDataType*>(c_g_m_n_device_buf.GetDeviceBuffer()),
                          M,
                          N,
                          K,
                          stride_A,
                          stride_B,
                          stride_C,
                          Scale_Stride_BN,
                          batch_stride_A,
                          batch_stride_B,
                          batch_stride_C,
                          batch_BScale_Stride,
                          static_cast<BScaleDataType*>(b1_g_scale_device_buf.GetDeviceBuffer()),
                          batch_count,
                          a_element_op,
                          b_element_op,
                          c_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return true;
    }

    std::size_t workspace_size = gemm.GetWorkSpaceSize(&argument);
    printf("workspace_size: %zu\n", workspace_size);

    bool pass = true;
    if(config.do_verification)
    {
        Tensor<float> b_g_k_n_dequant({batch_count, K, N});

        float v_b = 0;
        for(int bs = 0; bs < batch_count; bs++)
        {
            for(int n = 0; n < N; n++)
            {
                for(int k = 0; k < K; k++)
                {
                    ck::pk_i4_t i4x2 = b_g_k_n(bs, k, n);
                    int8_t i4        = 0;
                    if(k % 2 == 1)
                        i4 = (i4x2 >> 0) & 0xf;
                    else
                        i4 = (i4x2 >> 4) & 0xf;
                    i4  = i4 - 8;
                    v_b = ck::type_convert<float>(i4);

                    b_g_k_n_dequant(bs, k, n) =
                        ck::type_convert<float>(v_b) *
                        ck::type_convert<float>(b1_g_k_n(bs, k / Scale_Block_K, n / Scale_Block_N));

                    // printf("b_g_k_n_dequant(%d, %d, %d): %f\n", bs, k, n, b_g_k_n_dequant(bs, k, n));
                }
            }
        }

        auto ref_gemm    = ReferenceBatchedGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(a_g_m_k,
                                                  b_g_k_n_dequant,
                                                  c_g_m_n_host_result,
                                                  PassThrough{},
                                                  PassThrough{},
                                                  PassThrough{});

        ref_invoker.Run(ref_argument);

        ave_time = invoker.Run(argument, StreamConfig{nullptr, false, 0});
        c_g_m_n_device_buf.FromDevice(c_g_m_n_device_result.mData.data());

        pass &= ck::utils::check_err(c_g_m_n_device_result,
                                     c_g_m_n_host_result,
                                     "Error: Incorrect results!",
                                     get_rtol<CDataType>(),
                                     get_atol<CDataType>());

#if 0
        std::cout << "a_m_k: " << std::endl;
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < K; j++)
            {
                std::cout << ck::type_convert<float>(a_m_k(i, j)) << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "b_k_n: " << std::endl;
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < K; j++)
            {
                ck::pk_i4_t i4x2 = b_k_n(j, i);
                int8_t i4 = 0;
                if( j % 2 == 1)
                    i4 = (i4x2 >> 0) & 0xf;
                else
                    i4 = (i4x2 >> 4) & 0xf;
                i4 = i4 - 8;
                std::cout << ck::type_convert<float>(i4) << ",";
            }
            std::cout << std::endl;
        }

        std::cout<<"scale_b1_k_n: "<<std::endl;
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < (K + Scale_Block_K - 1) / Scale_Block_K; j++)
            {
                std::cout << ck::type_convert<float>(b1_k_n(j,i)) << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "c_m_n_device_result: " << std::endl;
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                std::cout << ck::type_convert<float>(c_m_n_device_result(i, j)) << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "c_m_n_host_result: " << std::endl;
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                std::cout << ck::type_convert<float>(c_m_n_host_result(i, j)) << ",";
            }
            std::cout << std::endl;
        }
#endif
    }

    if(config.time_kernel)
    {
        ave_time =
            invoker.Run(argument, StreamConfig{nullptr, config.time_kernel, 0, 20, 50, true, 50});

        std::size_t flop = 2_uz * M * N * K;
        std::size_t num_btype =
            sizeof(ADataType) * M * K +
            sizeof(BDataType) * K * N /
                (ck::is_same_v<ck::remove_cvref_t<BDataType>, ck::pk_i4_t> ? 2 : 1) +
            sizeof(CDataType) * M * N;

        float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

        float gb_per_sec = num_btype / 1.E6 / ave_time;

        std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                  << " GB/s, " << gemm.GetTypeString() << std::endl;
    }
    return pass;
}

bool run_batched_gemm_example(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    // std::mt19937 gen(11939);
    // std::uniform_int_distribution<int> dis(0, 15);

    // problem_size.M = 256 * (dis(gen) + 1);
    // problem_size.N = 128 * (dis(gen) + 1);
    // problem_size.K = 64 * (dis(gen) + 2);

    problem_size.stride_A = problem_size.K;
    problem_size.stride_B = problem_size.K;
    problem_size.stride_C = problem_size.N;

    problem_size.batch_stride_A = problem_size.M * problem_size.K;
    problem_size.batch_stride_B = problem_size.K * problem_size.N;
    problem_size.batch_stride_C = problem_size.M * problem_size.N;

    problem_size.batch_count = 2;

    if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else if(argc >= 10)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        problem_size.M = std::stoi(argv[4]);
        problem_size.N = std::stoi(argv[5]);
        problem_size.K = std::stoi(argv[6]);

        problem_size.stride_A = std::stoi(argv[7]);
        problem_size.stride_B = std::stoi(argv[8]);
        problem_size.stride_C = std::stoi(argv[9]);

        if(argc >= 11)
        {
            problem_size.batch_count = std::stoi(argv[10]);
        }
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        exit(0);
    }

    return run_batched_gemm(problem_size, config);
}

int main(int argc, char* argv[]) { return !run_batched_gemm_example(argc, argv); }
