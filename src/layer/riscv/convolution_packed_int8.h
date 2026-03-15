// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Ref: src/layer/arm/convolution_packed_int8.h
//      src/layer/x86/convolution_packed_int8.h
static void convolution_transform_kernel_packed_int8(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb

    // clang-format off
    // *INDENT-OFF*
#if __riscv_vector
    const size_t packn = csrr_vlenb();
    const int papb = packn / 2;
    if (outch >= papb)
    {
        if (inch >= papb)
            kernel_tm.create(maxk, inch / papb + inch % papb, outch / papb + outch % papb, (size_t)(papb * papb), papb * papb);
        else
            kernel_tm.create(maxk, inch, outch / papb + outch % papb, (size_t)papb, papb);
    }
    else
#endif // __riscv_vector
    {
#if __riscv_vector
        if (inch >= papb)
            kernel_tm.create(maxk, inch / papb + inch % papb, outch, (size_t)papb, papb);
        else
#endif // __riscv_vector
            kernel_tm.create(maxk, inch, outch, (size_t)1u, 1);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __riscv_vector
    for (; q + papb - 1 < outch; q += papb)
    {
        signed char* g00 = kernel_tm.channel(q / papb);

        int p = 0;
        for (; p + papb - 1 < inch; p += papb)
        {
            for (int k = 0; k < maxk; k++)
            {
                size_t vl = papb;
                ptrdiff_t stride_bytes = inch * maxk;
                for (size_t i = 0; i < vl; i++)
                {
                    // Perform (papb, papb) transpose
                    // TODO: Verify this in playground
                    const signed char* src = (const signed char*)kernel + q * inch * maxk + (p + i) * maxk + k;
                    vint8m1_t row = __riscv_vlse8_v_i8m1(src, stride_bytes, vl);
                    __riscv_vse8_v_i8m1(g00 + i * vl, row, vl);
                }
                g00 += (papb * papb);
            }
        }
        for (; p < inch; p++)
        {
            // TODO
        }
    }
#endif // __riscv_vector
    for (; q < outch; q++)
    {
        int p = 0;
#if __riscv_vector
        for (; p + papb - 1 < inch; p += papb)
        {
            // TODO
        }
#endif // __riscv_vector
        for (; p < inch; p++)
        {
            // TODO
        }
    }
    return;
}

static void convolution_packed_int8(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    return;
}