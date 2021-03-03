#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_dynamic_convolution_forward_implicit_gemm_v4r4_nhwc_kyxc_nhwk.hpp"

template <class T,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads>
void device_dynamic_convolution_forward_implicit_gemm_v4r4_nhwc_kyxc_nhwk(InDesc,
                                                                          const Tensor<T>& in_nchw,
                                                                          WeiDesc,
                                                                          const Tensor<T>& wei_kcyx,
                                                                          OutDesc,
                                                                          Tensor<T>& out_nkhw,
                                                                          ConvStrides,
                                                                          ConvDilations,
                                                                          InLeftPads,
                                                                          InRightPads,
                                                                          ck::index_t nrepeat)
{
    std::cout << "device_dynamic_convolution_forward_implicit_gemm_v4r4_nhwc_kyxc_nhwk"
              << std::endl;

    using namespace ck;

    using TDevice = typename conditional<is_same<half_float::half, T>::value, half_t, T>::type;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto N = OutDesc::GetLengths()[I0];
    constexpr auto K = OutDesc::GetLengths()[I1];
    constexpr auto C = WeiDesc::GetLengths()[I1];

    constexpr auto Hi = InDesc::GetLengths()[I2];
    constexpr auto Wi = InDesc::GetLengths()[I3];

    constexpr auto Ho = OutDesc::GetLengths()[I2];
    constexpr auto Wo = OutDesc::GetLengths()[I3];

    constexpr auto Y = WeiDesc::GetLengths()[I2];
    constexpr auto X = WeiDesc::GetLengths()[I3];

#if 0
    // run-time variables
    constexpr auto in_n_hi_wi_c_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_multi_index(N, Hi, Wi, C));
    constexpr auto wei_k_y_x_c_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_multi_index(K, Y, X, C));
    constexpr auto out_n_ho_wo_k_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_multi_index(N, Ho, Wo, K));

    const auto conv_strides   = to_multi_index(ConvStrides{});
    const auto conv_dilations = to_multi_index(ConvDilations{});
    const auto in_left_pads   = to_multi_index(InLeftPads{});
    const auto in_right_pads  = to_multi_index(InRightPads{});
#else
    // compile-time variables
    constexpr auto in_n_hi_wi_c_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, Hi, Wi, C));
    constexpr auto wei_k_y_x_c_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(K, Y, X, C));
    constexpr auto out_n_ho_wo_k_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, Ho, Wo, K));

    const auto conv_strides   = sequence_to_tuple_of_number(ConvStrides{});
    const auto conv_dilations = sequence_to_tuple_of_number(ConvDilations{});
    const auto in_left_pads   = sequence_to_tuple_of_number(InLeftPads{});
    const auto in_right_pads  = sequence_to_tuple_of_number(InRightPads{});
#endif

    Tensor<float> in_nhwc(
        make_HostTensorDescriptor(make_native_tensor_descriptor_packed(Sequence<N, Hi, Wi, C>{})));
    Tensor<float> wei_kyxc(
        make_HostTensorDescriptor(make_native_tensor_descriptor_packed(Sequence<K, Y, X, C>{})));
    Tensor<float> out_nhwk(
        make_HostTensorDescriptor(make_native_tensor_descriptor_packed(Sequence<N, Ho, Wo, K>{})));

    auto f_nchw2nhwc = [&](auto n, auto hi, auto wi, auto c) {
        in_nhwc(n, hi, wi, c) = in_nchw(n, c, hi, wi);
    };

    auto f_kcyx2kyxc = [&](auto k, auto y, auto x, auto c) {
        wei_kyxc(k, y, x, c) = wei_kcyx(k, c, y, x);
    };

    auto f_nkhw2nhwk = [&](auto n, auto ho, auto wo, auto k) {
        out_nhwk(n, ho, wo, k) = out_nkhw(n, k, ho, wo);
    };

    make_ParallelTensorFunctor(f_nchw2nhwc, N, Hi, Wi, C)(std::thread::hardware_concurrency());
    make_ParallelTensorFunctor(f_kcyx2kyxc, K, Y, X, C)(std::thread::hardware_concurrency());
    make_ParallelTensorFunctor(f_nkhw2nhwk, N, Ho, Wo, K)(std::thread::hardware_concurrency());

    std::size_t data_sz = sizeof(T);

    DeviceMem in_nhwc_device_buf(data_sz * in_nhwc.mDesc.GetElementSpace());
    DeviceMem wei_kyxc_device_buf(data_sz * wei_kyxc.mDesc.GetElementSpace());
    DeviceMem out_nhwk_device_buf(data_sz * out_nhwk.mDesc.GetElementSpace());

    in_nhwc_device_buf.ToDevice(in_nhwc.mData.data());
    wei_kyxc_device_buf.ToDevice(wei_kyxc.mData.data());
    out_nhwk_device_buf.ToDevice(out_nhwk.mData.data());

#if 1
    // cdata = 16, BlockSize = 64, 16x64x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 2;
    constexpr index_t GemmNPerThread = 2;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 2;
    constexpr index_t ThreadGemmDataPerReadN = 2;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 2;
#elif 0
    // cdata = 32, BlockSize = 64, 16x128x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 2;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 2;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 2>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 2;
#elif 0
    // cdata = 64, BlockSize = 64, 16x256x2
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 1;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 16;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<2, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<2, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 2;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 4;
#elif 0
    // cdata = 64, BlockSize = 64, 16x256x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 1;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 16;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 4;
#elif 0
    // cdata = 64, BlockSize = 128, 32x256x4
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 32;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 16;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 32>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 2>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 4;
#elif 0
    // cdata = 64, BlockSize = 128, 32x256x8
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 32;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 16;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<2, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 32>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 2;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<8, 2>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 8;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 4;
#elif 0
    // cdata = 64, BlockSize = 256, 128x128x8
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 8;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 4;
#elif 1
    // cdata = 64, BlockSize = 256, 128x128x16
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 16;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 8;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<4, 2>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 64>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 2;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<8, 1>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 8;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 4;
#endif

    constexpr auto conv_driver =
#if 1
        DriverDynamicConvolutionForwardImplicitGemm_v4r4_nhwc_kyxc_nhwk_pad
#elif 0
        DriverDynamicConvolutionForwardImplicitGemm_v4r4_nhwc_kyxc_nhwk_no_pad
#elif 1
        DriverDynamicConvolutionForwardImplicitGemm_v4r4_nhwc_kyxc_nhwk_1x1
#endif
        <BlockSize,
         TDevice,
         TDevice,
         GemmMPerBlock,
         GemmNPerBlock,
         GemmKPerBlock,
         GemmMPerThread,
         GemmNPerThread,
         GemmKPerThread,
         GemmMLevel0Cluster,
         GemmNLevel0Cluster,
         GemmMLevel1Cluster,
         GemmNLevel1Cluster,
         GemmABlockTransferThreadSliceLengths_GemmK_GemmM,
         GemmABlockTransferThreadClusterLengths_GemmK_GemmM,
         GemmABlockTransferSrcScalarPerVector_GemmK,
         GemmABlockTransferDstScalarPerVector_GemmM,
         GemmBBlockTransferThreadSliceLengths_GemmK_GemmN,
         GemmBBlockTransferThreadClusterLengths_GemmK_GemmN,
         GemmBBlockTransferSrcScalarPerVector_GemmK,
         GemmBBlockTransferDstScalarPerVector_GemmN,
         GemmCThreadTransferDstScalarPerVector_GemmM1>{};

    conv_driver.Run(wei_k_y_x_c_desc,
                    in_n_hi_wi_c_desc,
                    out_n_ho_wo_k_desc,
                    conv_strides,
                    conv_dilations,
                    in_left_pads,
                    in_right_pads,
                    static_cast<TDevice*>(wei_kyxc_device_buf.GetDeviceBuffer()),
                    static_cast<TDevice*>(in_nhwc_device_buf.GetDeviceBuffer()),
                    static_cast<TDevice*>(out_nhwk_device_buf.GetDeviceBuffer()));

    out_nhwk_device_buf.FromDevice(out_nhwk.mData.data());

    auto f_nhwk2nkhw = [&](auto n, auto k, auto ho, auto wo) {
        out_nkhw(n, k, ho, wo) = out_nhwk(n, ho, wo, k);
    };

    make_ParallelTensorFunctor(f_nhwk2nkhw, N, K, Ho, Wo)(std::thread::hardware_concurrency());
}
