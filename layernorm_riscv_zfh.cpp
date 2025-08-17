// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layernorm_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH && __riscv_zvfh
static inline int layernorm_rvv_pack1_fp16s_procedure(int size, __fp16* ptr, const float* gamma_data, const float* beta_data, float eps, int affine)
{
    float sum = 0.f;
    float sqsum = 0.f;
    vfloat32m1_t _sum = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
    vfloat32m1_t _sqsum = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
    {
        int n = size;
        __fp16* ptr_sum = ptr;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);
            vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr_sum, vl), vl);
            _sum = __riscv_vfredusum_vs_f32m8_f32m1(_p, /* scalar */ _sum, vl);
            // _sqsum = vfredusum_vs_f32m8_f32m1(_sqsum, vfmul_vv_f32m8(_p, _p, vl), /* scalar */ _sqsum, vl);
            ptr_sum += vl;
            n -= vl;
        }
    }
    sum = __riscv_vfmv_f_s_f32m1_f32(_sum);
    float mean = sum / size;

    {
        int n = size;
        __fp16* ptr_sqsum = ptr;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);
            vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr_sqsum, vl), vl);
            _p = __riscv_vfsub_vf_f32m8(_p, mean, vl);
            _sqsum = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(_p, _p, vl), /* scalar */ _sqsum, vl);
            n -= vl;
            ptr_sqsum += vl;
        }
    }
    sqsum = __riscv_vfmv_f_s_f32m1_f32(_sqsum);
    float var = sqsum / size;
    // the var maybe minus due to accuracy
    //float var = sqsum / size - mean * mean;
    float a = static_cast<float>(1.f / (sqrt(var + eps)));
    float b = -mean * a;

    {
        int n = size;
        __fp16* ptr_store = ptr;
        const float* ptr_gamma = gamma_data;
        const float* ptr_beta = beta_data;
        if (affine)
        {
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m4(n);
                vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr_store, vl), vl);
                _p = __riscv_vfmul_vf_f32m8(_p, a, vl);
                vfloat32m8_t _gamma = __riscv_vle32_v_f32m8(ptr_gamma, vl);
                _p = __riscv_vfadd_vf_f32m8(_p, b, vl);
                vfloat32m8_t _beta = __riscv_vle32_v_f32m8(ptr_beta, vl);
                _p = __riscv_vfmadd_vv_f32m8(_p, _gamma, _beta, vl);
                __riscv_vse16_v_f16m4(ptr_store, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);

                n -= vl;
                ptr_store += vl;
                ptr_gamma += vl;
                ptr_beta += vl;
            }
        }
        else
        {
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m4(n);
                vfloat32m8_t _p = __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4(ptr_store, vl), vl);
                _p = __riscv_vfmul_vf_f32m8(_p, a, vl);
                _p = __riscv_vfadd_vf_f32m8(_p, b, vl);
                __riscv_vse16_v_f16m4(ptr_store, __riscv_vfncvt_f_f_w_f16m4(_p, vl), vl);
                n -= vl;
                ptr_store += vl;
            }
        }
    }
    return 0;
}

static inline int layernorm_rvv_packn_fp16s_procedure(int size, __fp16* ptr, const float* gamma_data, const float* beta_data, float eps, int affine, const size_t vl)
{
    // mean and var
    // f16m1 => f32m2
    vfloat32m2_t _sum = __riscv_vfmv_v_f_f32m2(0.f, vl);
    vfloat32m2_t _sqsum = __riscv_vfmv_v_f_f32m2(0.f, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr + vl * i, vl), vl);
        _sum = __riscv_vfadd_vv_f32m2(_p, _sum, vl);
    }
    vfloat32m2_t _mean = __riscv_vfdiv_vf_f32m2(_sum, (float)size, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr + vl * i, vl), vl);
        _p = __riscv_vfsub_vv_f32m2(_p, _mean, vl);
        _sqsum = __riscv_vfmacc_vv_f32m2(_sqsum, _p, _p, vl);
    }
    vfloat32m2_t _var = __riscv_vfdiv_vf_f32m2(_sqsum, (float)size, vl);

    // the var maybe minus due to accuracy
    //float var = sqsum / size - mean * mean;
    vfloat32m2_t _a = __riscv_vfrdiv_vf_f32m2(__riscv_vfsqrt_v_f32m2(__riscv_vfadd_vf_f32m2(_var, eps, vl), vl), 1.f, vl);
    // how about vfrsqrt7.v?
    vfloat32m2_t _b = __riscv_vfmul_vv_f32m2(__riscv_vfsgnjn_vv_f32m2(_mean, _mean, vl), _a, vl);
    if (affine)
    {
        for (int i = 0; i < size; i++)
        {
            const int offset = vl * i;
            vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr + offset, vl), vl);
            _p = __riscv_vfmadd_vv_f32m2(_p, _a, _b, vl);
            _p = __riscv_vfmul_vf_f32m2(_p, gamma_data[i], vl);
            _p = __riscv_vfadd_vf_f32m2(_p, beta_data[i], vl);
            __riscv_vse16_v_f16m1(ptr + offset, __riscv_vfncvt_f_f_w_f16m1(_p, vl), vl);
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            const int offset = vl * i;
            vfloat32m2_t _p = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr + offset, vl), vl);
            _p = __riscv_vfmadd_vv_f32m2(_p, _a, _b, vl);
            __riscv_vse16_v_f16m1(ptr + offset, __riscv_vfncvt_f_f_w_f16m1(_p, vl), vl);
        }
    }

    return 0;
}

static inline int layernorm_rvv_pack1_fp16sa_procedure(int size, __fp16* ptr, const float* gamma_data, const float* beta_data, float eps, int affine)
{
    __fp16 sum = 0.f;
    __fp16 sqsum = 0.f;
    vfloat16m1_t _sum = __riscv_vfmv_s_f_f16m1(0.f, __riscv_vsetvlmax_e32m1());
    vfloat16m1_t _sqsum = __riscv_vfmv_s_f_f16m1(0.f, __riscv_vsetvlmax_e32m1());
    {
        int n = size;
        __fp16* ptr_sum = ptr;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m8(n);
            vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr_sum, vl);
            _sum = __riscv_vfredusum_vs_f16m8_f16m1(_p, /* scalar */ _sum, vl);
            // _sqsum = vfredusum_vs_f32m8_f32m1(_sqsum, vfmul_vv_f32m8(_p, _p, vl), /* scalar */ _sqsum, vl);
            ptr_sum += vl;
            n -= vl;
        }
    }
    sum = __riscv_vfmv_f_s_f16m1_f16(_sum);
    __fp16 mean = sum / size;

    {
        int n = size;
        __fp16* ptr_sqsum = ptr;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m8(n);
            vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr_sqsum, vl);
            _p = __riscv_vfsub_vf_f16m8(_p, mean, vl);
            _sqsum = __riscv_vfredusum_vs_f16m8_f16m1(__riscv_vfmul_vv_f16m8(_p, _p, vl), /* scalar */ _sqsum, vl);
            n -= vl;
            ptr_sqsum += vl;
        }
    }
    sqsum = __riscv_vfmv_f_s_f16m1_f16(_sqsum);
    __fp16 var = sqsum / size;
    // the var maybe minus due to accuracy
    //float var = sqsum / size - mean * mean;
    __fp16 a = static_cast<__fp16>(1.f / (sqrt(var + eps)));
    __fp16 b = static_cast<__fp16>(-mean * a);

    {
        int n = size;
        __fp16* ptr_store = ptr;
        const float* ptr_gamma = gamma_data;
        const float* ptr_beta = beta_data;
        if (affine)
        {
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m4(n);
                vfloat16m4_t _p = __riscv_vle16_v_f16m4(ptr_store, vl);
                _p = __riscv_vfmul_vf_f16m4(_p, a, vl);
                vfloat16m4_t _gamma = __riscv_vfncvt_f_f_w_f16m4(__riscv_vle32_v_f32m8(ptr_gamma, vl), vl);
                _p = __riscv_vfadd_vf_f16m4(_p, b, vl);
                vfloat16m4_t _beta = __riscv_vfncvt_f_f_w_f16m4(__riscv_vle32_v_f32m8(ptr_beta, vl), vl);
                _p = __riscv_vfmadd_vv_f16m4(_p, _gamma, _beta, vl);
                __riscv_vse16_v_f16m4(ptr_store, _p, vl);

                n -= vl;
                ptr_store += vl;
                ptr_gamma += vl;
                ptr_beta += vl;
            }
        }
        else
        {
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m8(n);
                vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr_store, vl);
                _p = __riscv_vfmul_vf_f16m8(_p, a, vl);
                _p = __riscv_vfadd_vf_f16m8(_p, b, vl);
                __riscv_vse16_v_f16m8(ptr_store, _p, vl);
                n -= vl;
                ptr_store += vl;
            }
        }
    }
    return 0;
}

static inline int layernorm_rvv_packn_fp16sa_procedure(int size, __fp16* ptr, const float* gamma_data, const float* beta_data, float eps, int affine, const size_t vl)
{
    // mean and var
    vfloat16m1_t _sum = __riscv_vfmv_v_f_f16m1(0.f, vl);
    vfloat16m1_t _sqsum = __riscv_vfmv_v_f_f16m1(0.f, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat16m1_t _p = __riscv_vle16_v_f16m1(ptr + vl * i, vl);
        _sum = __riscv_vfadd_vv_f16m1(_p, _sum, vl);
        // _sqsum = vfmadd_vv_f16m1(_p,_p,_sqsum,vl);
    }
    vfloat16m1_t _mean = __riscv_vfdiv_vf_f16m1(_sum, size, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat16m1_t _p = __riscv_vle16_v_f16m1(ptr + vl * i, vl);
        _p = __riscv_vfsub_vv_f16m1(_p, _mean, vl);
        _sqsum = __riscv_vfmacc_vv_f16m1(_sqsum, _p, _p, vl);
    }
    vfloat16m1_t _var = __riscv_vfdiv_vf_f16m1(_sqsum, size, vl);

    // the var maybe minus due to accuracy
    //float var = sqsum / size - mean * mean;
    vfloat16m1_t _a = __riscv_vfrdiv_vf_f16m1(__riscv_vfsqrt_v_f16m1(__riscv_vfadd_vf_f16m1(_var, eps, vl), vl), 1.f, vl);
    // how about vfrsqrt7.v?
    vfloat16m1_t _b = __riscv_vfmul_vv_f16m1(__riscv_vfsgnjn_vv_f16m1(_mean, _mean, vl), _a, vl);
    if (affine)
    {
        for (int i = 0; i < size; i++)
        {
            const int offset = vl * i;
            vfloat16m1_t _p = __riscv_vle16_v_f16m1(ptr + offset, vl);
            _p = __riscv_vfmadd_vv_f16m1(_p, _a, _b, vl);
            _p = __riscv_vfmul_vf_f16m1(_p, gamma_data[i], vl);
            _p = __riscv_vfadd_vf_f16m1(_p, beta_data[i], vl);
            __riscv_vse16_v_f16m1(ptr + offset, _p, vl);
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            const int offset = vl * i;
            vfloat16m1_t _p = __riscv_vle16_v_f16m1(ptr + offset, vl);
            _p = __riscv_vfmadd_vv_f16m1(_p, _a, _b, vl);
            __riscv_vse16_v_f16m1(ptr + offset, _p, vl);
        }
    }

    return 0;
}
#else
static inline int layernorm_scaler_fp16s_procedure(int size, __fp16* ptr, const float* gamma_data, const float* beta_data, float eps, int affine)
{
    __fp16 sum = 0.f, sqsum = 0.f, *gamma_data_fp16 = (__fp16*)gamma_data, *beta_data_fp16 = (__fp16*)beta_data;

    for (int i = 0; i < size; i++) sum += ptr[i];

    __fp16 mean = sum / size;
    __fp16 tmp = 0.f;
    for (int i = 0; i < size; i++)
    {
        tmp = ptr[i] - mean;
        sqsum += tmp * tmp;
    }

    __fp16 var = sqsum / size;
    __fp16 a = static_cast<__fp16>(1.f / (sqrt(var + eps)));
    __fp16 b = -mean * a;

    if (affine)
        for (int i = 0; i < size; i++) ptr[i] = (ptr[i] * a + b) * gamma_data[i] + beta_data[i];
    else
        for (int i = 0; i < size; i++) ptr[i] = ptr[i] * a + b;

    return 0;
}
#endif // NCNN_ZFH && __riscv_zvfh

int LayerNorm_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    // x = (x - mean) / sqrt(var + eps) * gamma + beta
    int elempack = bottom_top_blob.elempack;
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;

    if (dims == 1)
    {
        __fp16* ptr = bottom_top_blob;
#if __riscv_zvfh
        return layernorm_rvv_pack1_fp16s_procedure(w * elempack, ptr, gamma_data, beta_data, eps, affine);
#else
        return layernorm_scaler_fp16s_procedure(w, ptr, gamma_data, beta_data, eps, affine);
#endif
    }

#if __riscv_zvfh
    if (elempack == 1)
#endif  // __riscv_zvfh
    {
        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            // assert affine_size == w
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                
#if __riscv_zvfh
                layernorm_rvv_pack1_fp16s_procedure(w, ptr, gamma_data, beta_data, eps, affine);
#else
                layernorm_scaler_fp16s_procedure(w, ptr, gamma_data, beta_data, eps, affine);
#endif
            }
        }

        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;
            if (affine_size == w)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        __fp16* ptr = bottom_top_blob.channel(q).row<__fp16>(i);
#if __riscv_zvfh
                        layernorm_rvv_pack1_fp16s_procedure(w, ptr, gamma_data, beta_data, eps, affine);
#else
                        layernorm_scaler_fp16s_procedure(w, ptr, gamma_data, beta_data, eps, affine);
#endif
                
                    }
                }
            }
            else // if (affine_size == size)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    __fp16* ptr = bottom_top_blob.channel(q);
#if __riscv_zvfh
                    layernorm_rvv_pack1_fp16s_procedure(size, ptr, gamma_data, beta_data, eps, affine);
#else
                    layernorm_scaler_fp16s_procedure(size, ptr, gamma_data, beta_data, eps, affine);
#endif
                }
            }
        }

        return 0;
    }

    const int packn = csrr_vlenb() / 2; // fp16
    if (elempack == packn)
    {
        const size_t vl = __riscv_vsetvl_e16m1(packn);
        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            // assert affine_size == w

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __fp16* ptr = bottom_top_blob.row<__fp16>(i);
                layernorm_rvv_packn_fp16s_procedure(w, ptr, gamma_data, beta_data, eps, affine, vl);
            }
        }
        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            if (affine_size == w)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        __fp16* ptr = bottom_top_blob.channel(q).row<__fp16>(i);
                        layernorm_rvv_packn_fp16s_procedure(w, ptr, gamma_data, beta_data, eps, affine, vl);
                    }
                }
            }
            else // if (affine_size == size)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    __fp16* ptr = bottom_top_blob.channel(q);
                    layernorm_rvv_packn_fp16s_procedure(size, ptr, gamma_data, beta_data, eps, affine, vl);
                }
            }
        }
    }

    return 0;
}
#endif // NCNN_ZFH && __riscv_zvfh
} // namespace ncnn
