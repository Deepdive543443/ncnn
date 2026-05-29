// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Ref: src/layer/arm/convolution_packed_int8.h
//      src/layer/x86/convolution_packed_int8.h

static void convolution_transform_kernel_packed_int8_rvv(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/vlm1-outch/vlm1

    // clang-format off
    // *INDENT-OFF*
#if __riscv_vector
    const int packn_s8 = csrr_vlenb();
    const int packn = csrr_vlenb() / 4;

    if (outch >= packn_s8)
    {
        if (inch >= packn_s8)
            kernel_tm.create(maxk, inch / packn_s8 + inch % packn_s8, outch / packn_s8 + (outch % packn_s8) / packn + outch % packn, (size_t)(packn_s8 * packn_s8), packn_s8 * packn_s8);
        else
            kernel_tm.create(maxk, inch, outch / packn_s8 + (outch % packn_s8) / packn + outch % packn, (size_t)packn_s8, packn_s8);
    }
    else if (outch >= packn)
    {
        if (inch >= packn_s8)
            kernel_tm.create(maxk, inch / packn_s8 + inch % packn_s8, outch / packn + outch % packn, (size_t)(packn_s8 * packn), packn_s8 * packn);
        else
            kernel_tm.create(maxk, inch, outch / packn + outch % packn, (size_t)packn, packn);
    }
    else
#endif // __riscv_vector
    {
#if __riscv_vector
        if (inch >= packn_s8)
            kernel_tm.create(maxk, inch / packn_s8 + inch % packn_s8, outch, (size_t)packn_s8, packn_s8);
        else
#endif // __riscv_vector
            kernel_tm.create(maxk, inch, outch, (size_t)1u, 1);
    }
    // *INDENT-ON*
    // clang-format on

    int q = 0;
#if __riscv_vector
    for (; q + packn_s8 - 1 < outch; q += packn_s8)
    {
        const signed char* kptr = (const signed char*)kernel + q * inch * maxk;
        signed char* g00 = kernel_tm.channel(q / packn_s8);

        int p = 0;
        for (; p + packn_s8 - 1 < inch; p += packn_s8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (size_t i = 0; i < packn_s8; i++)
                {
                    const signed char* src = kptr + (p + i) * maxk + k;
                    vint8m1_t row = __riscv_vlse8_v_i8m1(src, inch * maxk, packn_s8);
                    __riscv_vse8_v_i8m1(g00, row, packn_s8);
                    g00 += packn_s8;
                }
            }
        }
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* src = kptr + p * maxk + k;
                vint8m1_t row = __riscv_vlse8_v_i8m1(src, inch * maxk, packn_s8);
                __riscv_vse8_v_i8m1(g00, row, packn_s8);
                g00 += packn_s8;
            }
        }
    }
    for (; q + packn - 1 < outch; q += packn)
    {
        const signed char* kptr = (const signed char*)kernel + q * inch * maxk;
        signed char* g00 = kernel_tm.channel(q / packn_s8 + (q % packn_s8) / packn);
        int p = 0;
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* src = kptr + p * maxk + k;
                vint8m1_t row = __riscv_vlse8_v_i8m1(src, inch * maxk, packn);
                __riscv_vse8_v_i8m1(g00, row, packn);
                g00 += packn;
            }
        }
    }
#endif // __riscv_vector
    for (; q < outch; q++)
    {
        const signed char* kptr = (const signed char*)kernel + q * inch * maxk;
#if __riscv_vector
        signed char* g00 = kernel_tm.channel(q / packn_s8 + (q % packn_s8) / packn + q % packn);
#else
        signed char* g00 = kernel_tm.channel(q);
#endif

        int p = 0;
#if __riscv_vector
        for (; p + packn_s8 - 1 < inch; p += packn_s8)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k0 = kptr + k;

                for (size_t i = 0; i < packn_s8; i++)
                {
                    g00[0] = k0[0];
                    k0 += maxk;
                    g00 += 1;
                }
            }
            kptr += maxk * packn_s8;
        }
#endif // __riscv_vector
        for (; p < inch; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                g00[0] = kptr[0];
                g00++;
                kptr++;
            }
        }
    }
    return;
}

static void convolution_packed_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    const int w = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int inch = bottom_blob.c * elempack;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int out_elempack = top_blob.elempack;
    const int outch = top_blob.c * out_elempack;
    const size_t N = bottom_blob.cstep * elempack;
    const size_t M = top_blob.cstep * out_elempack;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2 * elempack;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    int nn_outch = 0;
    int remain_outch_start = 0;
#if __riscv_vector
    const size_t vlm1 = __riscv_vsetvlmax_e8m1();
    const size_t packn_s8 = csrr_vlenb();
    const size_t packn = csrr_vlenb() / 4;

    nn_outch = (outch - remain_outch_start) / packn_s8;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const size_t vlm1 = __riscv_vsetvlmax_e8m1();

        const int p = remain_outch_start + pp * packn_s8;
        int* outptr = top_blob.channel(p / out_elempack);

        int ij = 0;
        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vlm1);
            const signed char* kptr = weight_data_tm.channel(p / packn_s8);

            int q = 0;
            for (; q + packn_s8 - 1 < inch; q += packn_s8)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    if (elempack == packn_s8)
                    {
                        for (int z = 0; z < packn_s8; z++)
                        {
                            vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vlm1);
                            vint16m2_t _s = __riscv_vwmul_vx_i16m2(_w, r0s[z], vlm1);
                            _sum = __riscv_vwadd_wv_i32m4(_sum, _s, vlm1);

                            kptr += packn_s8;
                        }
                    }
                    else
                    {
                        for (int z = 0; z < packn_s8; z++)
                        {
                            vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vlm1);
                            vint16m2_t _s = __riscv_vwmul_vx_i16m2(_w, r0s[z * N], vlm1);
                            _sum = __riscv_vwadd_wv_i32m4(_sum, _s, vlm1);

                            kptr += packn_s8;
                        }
                    }
                }
            }

            for (; q < inch; q++)
            {
                // If we reach here, bottom blob must unpacked
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;
                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vlm1);
                    vint16m2_t _s = __riscv_vwmul_vx_i16m2(_w, r0s[0], vlm1);
                    _sum = __riscv_vwadd_wv_i32m4(_sum, _s, vlm1);

                    kptr += packn_s8;
                }
            }

            if (out_elempack == packn)
            {
                __riscv_vse32_v_i32m1(outptr, __riscv_vget_v_i32m4_i32m1(_sum, 0), __riscv_vsetvlmax_e32m1());
                __riscv_vse32_v_i32m1(outptr + M, __riscv_vget_v_i32m4_i32m1(_sum, 1), __riscv_vsetvlmax_e32m1());
                __riscv_vse32_v_i32m1(outptr + M * 2, __riscv_vget_v_i32m4_i32m1(_sum, 2), __riscv_vsetvlmax_e32m1());
                __riscv_vse32_v_i32m1(outptr + M * 3, __riscv_vget_v_i32m4_i32m1(_sum, 3), __riscv_vsetvlmax_e32m1());
                outptr += packn;
            }

            if (out_elempack == 1)
            {
                __riscv_vsse32_v_i32m4(outptr, M * sizeof(int), _sum, vlm1);
                outptr += 1;
            }
        }
    }

    remain_outch_start += nn_outch * packn_s8;
    nn_outch = (outch - remain_outch_start) / packn;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        const size_t vl = __riscv_vsetvl_e8m1(packn);
        const int p = remain_outch_start + pp * packn;
        int* outptr = top_blob.channel(p / out_elempack);

        int ij = 0;
        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vl);
            const signed char* kptr = weight_data_tm.channel(p / packn_s8 + (p % packn_s8) / packn);
            int q = 0;
            for (; q + packn_s8 - 1 < inch; q += packn_s8)
            {
                const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                if (elempack == packn_s8)
                {
                    for (int z = 0; z < packn_s8; z++)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const signed char* r0s = r0 + space_ofs[k];
                            vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vl);
                            vint16m2_t _s = __riscv_vwmul_vx_i16m2(_w, r0s[z], vl);
                            _sum = __riscv_vwadd_wv_i32m4(_sum, _s, vl);

                            kptr += packn;
                        }
                    }
                }
                else
                {
                    for (int z = 0; z < packn_s8; z++)
                    {
                        for (int k = 0; k < maxk; k++)
                        {
                            const signed char* r0s = r0 + space_ofs[k];
                            vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vl);
                            vint16m2_t _s = __riscv_vwmul_vx_i16m2(_w, r0s[z * N], vl);
                            _sum = __riscv_vwadd_wv_i32m4(_sum, _s, vl);

                            kptr += vl;
                        }
                    }
                }
            }

            for (; q < inch; q++)
            {
                // If we reach here, bottom blob must unpacked
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;
                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vl);
                    vint16m2_t _s = __riscv_vwmul_vx_i16m2(_w, r0s[0], vl);
                    _sum = __riscv_vwadd_wv_i32m4(_sum, _s, vl);

                    kptr += vl;
                }
            }

            if (out_elempack == packn)
            {
                __riscv_vse32_v_i32m1(outptr, __riscv_vget_v_i32m4_i32m1(_sum, 0), vl);
                outptr += packn;
            }

            if (out_elempack == 1)
            {
                __riscv_vsse32_v_i32m4(outptr, M * sizeof(int), _sum, vl);
                outptr += 1;
            }
        }
    }

    remain_outch_start += nn_outch * packn;
#endif // __riscv_vector
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        int* outptr = top_blob.channel(p);

        int ij = 0;
        for (; ij < outw * outh; ij++)
        {
            const int i = ij / outw;
            const int j = ij % outw;

            int sum = 0;
#if __riscv_vector
            const signed char* kptr = weight_data_tm.channel(p / packn_s8 + (p % packn_s8) / packn + p % packn);
#else
            const signed char* kptr = weight_data_tm.channel(p);
#endif
            int q = 0;
#if __riscv_vector
            vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vlm1);

            if (elempack == packn_s8)
            {
                for (; q + vlm1 - 1 < inch; q += vlm1)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        vint8m1_t _r = __riscv_vle8_v_i8m1(r0s, vlm1);
                        vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vlm1);
                        vint16m2_t _s = __riscv_vwmul_vv_i16m2(_w, _r, vlm1);
                        _sum = __riscv_vwadd_wv_i32m4(_sum, _s, vlm1);
                        kptr += vlm1;
                    }
                }
            }
            else
            {
                for (; q + vlm1 - 1 < inch; q += vlm1)
                {
                    const signed char* r0 = bottom_blob.channel(q / elempack).row<const signed char>(i * stride_h) + j * stride_w * elempack;

                    for (int k = 0; k < maxk; k++)
                    {
                        const signed char* r0s = r0 + space_ofs[k];
                        vint8m1_t _r = __riscv_vlse8_v_i8m1(r0s, N, vlm1);
                        vint8m1_t _w = __riscv_vle8_v_i8m1(kptr, vlm1);
                        vint16m2_t _s = __riscv_vwmul_vv_i16m2(_w, _r, vlm1);
                        _sum = __riscv_vwadd_wv_i32m4(_sum, _s, vlm1);
                        kptr += vlm1;
                    }
                }
            }

            vint32m1_t _sum0 = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
            _sum0 = __riscv_vredsum_vs_i32m4_i32m1(_sum, _sum0, __riscv_vsetvlmax_e32m4());
            sum += __riscv_vmv_x_s_i32m1_i32(_sum0);
#endif
            for (; q < inch; q++)
            {
                const signed char* r0 = bottom_blob.channel(q).row<const signed char>(i * stride_h) + j * stride_w;
                for (int k = 0; k < maxk; k++)
                {
                    const signed char* r0s = r0 + space_ofs[k];
                    sum += r0s[0] * kptr[0];
                    kptr++;
                }
            }
            outptr[0] = sum;
            outptr += 1;
        }
    }

    return;
}