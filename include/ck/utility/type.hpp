// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/integral_constant.hpp"
#include "ck/utility/enable_if.hpp"

namespace ck {

template <typename X, typename Y>
struct is_same : public integral_constant<bool, false>
{
};

template <typename X>
struct is_same<X, X> : public integral_constant<bool, true>
{
};

template <typename X, typename Y>
inline constexpr bool is_same_v = is_same<X, Y>::value;

template <typename T>
using remove_reference_t = typename std::remove_reference<T>::type;

template <typename T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <typename T>
using remove_cvref_t = remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
using remove_pointer_t = typename std::remove_pointer<T>::type;

template <typename T>
inline constexpr bool is_pointer_v = std::is_pointer<T>::value;

template <typename Y, typename X, typename enable_if<sizeof(X) == sizeof(Y), bool>::type = false>
__host__ __device__ constexpr Y bit_cast(const X& x)
{
    static_assert(__has_builtin(__builtin_bit_cast), "");
    static_assert(sizeof(X) == sizeof(Y), "Do not support cast between different size of type");

    return __builtin_bit_cast(Y, x);
}

} // namespace ck
