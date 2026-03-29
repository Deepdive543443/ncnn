// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution_riscv.h"

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector
#include "riscv_activation.h"
#include "riscv_usability.h"

#include "cpu.h"

namespace ncnn {

#include "convolution_sgemm.h"
#include "convolution_winograd_transform.h"
#include "convolution_winograd_dot.h"
#include "convolution_1x1.h"
#include "convolution_3x3.h"

#if NCNN_INT8
#include "convolution_packed_int8.h"
#endif // NCNN_INT8

#if __riscv_vector
#include "convolution_packn.h"
#include "convolution_pack1ton.h"
#include "convolution_packnto1.h"

#include "convolution_sgemm_packn.h"
#include "convolution_sgemm_pack1ton.h"
#include "convolution_sgemm_packnto1.h"
#include "convolution_winograd_transform_packn.h"
#include "convolution_winograd_dot_packn.h"
#include "convolution_1x1_packn.h"
#include "convolution_1x1_pack1ton.h"
#include "convolution_1x1_packnto1.h"
#include "convolution_3x3_packn.h"
#include "convolution_3x3_pack1ton.h"
#include "convolution_7x7_pack1ton.h"
#endif // __riscv_vector

Convolution_riscv::Convolution_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif

    activation = 0;
}

static void convolution_transform_kernel_packed_rvv(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        weight_data_tm.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)4u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_tm.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        for (int j = 0; j < out_elempack; j++)
                        {
                            const float* k00 = weight_data_r2.channel(q + j).row(p + i);

                            g00[0] = k00[k];

                            g00++;
                        }
                    }
                }
            }
        }
    }
}

int Convolution_riscv::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return create_pipeline_int8_rvv(opt);
    }
#endif

#if NCNN_ZFH
    if (support_fp16_storage && opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
#endif

    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    int elempack = 1;
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        elempack = num_input % packn == 0 ? packn : 1;
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif

#if __riscv_vector
    // packn
    if (elempack == packn && out_elempack == packn)
    {
        if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if ((opt.use_winograd63_convolution && num_input >= packn * 2 && num_output >= packn * 2 && num_input <= packn * 16 && num_output <= packn * 16) || (!opt.use_winograd43_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd63_transform_kernel_packn_rvv(weight_data, weight_winograd63_data, num_input, num_output, opt);
            else if ((opt.use_winograd43_convolution && num_input >= packn * 2 && num_output >= packn * 2) || (!opt.use_winograd63_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd43_transform_kernel_packn_rvv(weight_data, weight_winograd43_data, num_input, num_output, opt);
            else // if (opt.use_winograd23_convolution)
                conv3x3s1_winograd23_transform_kernel_packn_rvv(weight_data, weight_winograd23_data, num_input, num_output, opt);
        }
        else
        {
            convolution_transform_kernel_packed_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    // pack1ton
    if (elempack == 1 && out_elempack == packn)
    {
        convolution_transform_kernel_packed_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
    }

    // packnto1
    if (elempack == packn && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_packnto1_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_packnto1_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_transform_kernel_packnto1_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            convolution_transform_kernel_packed_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }
#endif // __riscv_vector

    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if ((opt.use_winograd43_convolution && num_input >= 16 && num_output >= 16) || !opt.use_winograd23_convolution)
            {
                conv3x3s1_winograd43_transform_kernel_rvv(weight_data, weight_winograd43_data, num_input, num_output, opt);
            }
            else if (opt.use_winograd23_convolution)
            {
                conv3x3s1_winograd23_transform_kernel_rvv(weight_data, weight_winograd23_data, num_input, num_output, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_transform_kernel_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            weight_data_tm = weight_data;
        }
    }

    if (opt.lightmode)
        weight_data.release();

    return 0;
}

int Convolution_riscv::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    return 0;
}

int Convolution_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && int8_scale_term)
    {
        return forward_int8_rvv(bottom_blob, top_blob, opt);
    }
#endif

    // flattened blob, implement as InnerProduct
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        Mat bottom_blob_3d;
        if (bottom_blob.elemsize % 16 == 0)
        {
            bottom_blob_3d = bottom_blob;
            bottom_blob_3d.dims = 3;
            bottom_blob_3d.w = 1;
            bottom_blob_3d.h = 1;
            bottom_blob_3d.c = bottom_blob.w;
            bottom_blob_3d.cstep = 1;
        }
        else
        {
            bottom_blob_3d = bottom_blob.reshape(1, 1, bottom_blob.w, opt.workspace_allocator);
        }

        Mat top_blob_3d;
        int ret = forward(bottom_blob_3d, top_blob_3d, opt);
        if (ret != 0)
            return ret;

        if (top_blob_3d.elemsize % 16 == 0)
        {
            top_blob = top_blob_3d;
            top_blob.dims = 1;
            top_blob.w = top_blob_3d.c;
            top_blob.h = 1;
            top_blob.c = 1;
            top_blob.cstep = top_blob_3d.c;
        }
        else
        {
            top_blob = top_blob_3d.reshape(top_blob_3d.c, opt.blob_allocator);
        }

        return 0;
    }

#if NCNN_ZFH
    int elembits = bottom_blob.elembits();

    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    //     NCNN_LOGE("Convolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int num_input = channels * elempack;

#if __riscv_vector
    if (elempack == packn && out_elempack == packn)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if ((opt.use_winograd63_convolution && num_input >= packn * 2 && num_output >= packn * 2 && num_input <= packn * 16 && num_output <= packn * 16) || (!opt.use_winograd43_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd63_packn_rvv(bottom_blob_bordered, top_blob, weight_winograd63_data, bias_data, opt);
            else if ((opt.use_winograd43_convolution && num_input >= packn * 2 && num_output >= packn * 2) || (!opt.use_winograd63_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd43_packn_rvv(bottom_blob_bordered, top_blob, weight_winograd43_data, bias_data, opt);
            else // if (opt.use_winograd23_convolution)
                conv3x3s1_winograd23_packn_rvv(bottom_blob_bordered, top_blob, weight_winograd23_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 1 && out_elempack == packn)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv7x7s2_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == packn && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_packnto1_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_packnto1_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_packnto1_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_packnto1_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }
#endif // __riscv_vector

    if (elempack == 1 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_winograd_convolution && (opt.use_winograd43_convolution || opt.use_winograd23_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if ((opt.use_winograd43_convolution && num_input >= 16 && num_output >= 16) || !opt.use_winograd23_convolution)
            {
                conv3x3s1_winograd43_rvv(bottom_blob_bordered, top_blob, weight_winograd43_data, bias_data, opt);
            }
            else if (opt.use_winograd23_convolution)
            {
                conv3x3s1_winograd23_rvv(bottom_blob_bordered, top_blob, weight_winograd23_data, bias_data, opt);
            }

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
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
                        space_ofs[p1] = p2;
                        p1++;
                        p2 += dilation_w;
                    }
                    p2 += gap;
                }
            }

            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const float* kptr = (const float*)weight_data_tm + maxk * channels * p;

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = sptr[space_ofs[k]];
                                float wt = kptr[k];
                                sum += val * wt;
                            }

                            kptr += maxk;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

int Convolution_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _kernel_w = _weight_data.w;
    const int _kernel_h = _weight_data.h;
    const int _num_output = _weight_data.c * _weight_data.elempack;

    Mat weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

#if NCNN_ZFH
    if (opt.use_fp16_storage && support_fp16_storage && weight_data_flattened.elembits() == 16)
    {
        Mat weight_data_flattened_fp32;
        cast_float16_to_float32(weight_data_flattened, weight_data_flattened_fp32, opt);
        weight_data_flattened = weight_data_flattened_fp32;
    }
#endif // NCNN_ZFH

    // weight_data_flattened as pack1
    weight_data_flattened.w *= weight_data_flattened.elempack;
    weight_data_flattened.elemsize /= weight_data_flattened.elempack;
    weight_data_flattened.elempack = 1;

    Mat bias_data_flattened;
    if (bias_term)
    {
        const Mat& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;

#if NCNN_ZFH
        if (opt.use_fp16_storage && support_fp16_storage && bias_data_flattened.elembits() == 16)
        {
            Mat bias_data_flattened_fp32;
            cast_float16_to_float32(bias_data_flattened, bias_data_flattened_fp32, opt);
            bias_data_flattened = bias_data_flattened_fp32;
        }
#endif // NCNN_ZFH

        // bias_data_flattened as pack1
        bias_data_flattened.w *= bias_data_flattened.elempack;
        bias_data_flattened.elemsize /= bias_data_flattened.elempack;
        bias_data_flattened.elempack = 1;
    }

    ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Convolution);

    ncnn::ParamDict pd;
    pd.set(0, _num_output);
    pd.set(1, _kernel_w);
    pd.set(11, _kernel_h);
    pd.set(2, dilation_w);
    pd.set(12, dilation_h);
    pd.set(3, stride_w);
    pd.set(13, stride_h);
    pd.set(4, pad_left);
    pd.set(15, pad_right);
    pd.set(14, pad_top);
    pd.set(16, pad_bottom);
    pd.set(18, pad_value);
    pd.set(5, bias_term);
    pd.set(6, weight_data_flattened.w);
    pd.set(8, int8_scale_term);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    op->load_param(pd);

    ncnn::Mat weights[2];
    weights[0] = weight_data_flattened;
    weights[1] = bias_data_flattened;

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    op->forward(bottom_blob, top_blob, opt);

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

#if NCNN_INT8
int Convolution_riscv::create_pipeline_int8_rvv(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    // TODO: implment kernel transform for winograd, sgemm, etc
    convolution_transform_kernel_packed_int8_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);

    scale_in_data.create(num_output);
    for (int p = 0; p < num_output; p++)
    {
        // requantize and relu
        float scale_in;
        if (weight_data_int8_scales[p] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);

        scale_in_data[p] = scale_in;
    }

    if (opt.lightmode)
        weight_data.release();

    return 0;
}

int Convolution_riscv::forward_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __riscv_vector
    const int packn_32 = csrr_vlenb() / 4; // int32
    const int packn = csrr_vlenb();        // requantized int8
#endif                                     // __riscv_vector

    int elembits = bottom_blob.elembits();

    Mat bottom_blob_int8 = bottom_blob;
    if (elembits != 8)
    {
        Option opt_q = opt;
        opt_q.blob_allocator = opt.workspace_allocator;
#if NCNN_ZFH
        if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        {
            Mat bottom_blob_fp32;
            cast_float16_to_float32(bottom_blob, bottom_blob_fp32, opt);
            quantize_to_int8(bottom_blob_fp32, bottom_blob_int8, bottom_blob_int8_scales, opt_q);
            fprintf(stderr, "input (fp16): \n");
            pretty_print<float>(bottom_blob_fp32);
        }
        else
#endif
        {
            fprintf(stderr, "input (fp32): \n");
            pretty_print<float>(bottom_blob);

            quantize_to_int8(bottom_blob, bottom_blob_int8, bottom_blob_int8_scales, opt_q);
            // Native quantize_to_int8 didn't handle packed input properply, lead to unexpected output

            // Print from unit test
            // input (fp32):
            // shape: 9 x 7 x (1 pack 8)
            // 0.631522 -0.921565 0.809960 0.784604 0.879260 -0.688094 -1.186034 0.155285 -0.000831 0.478068 1.102549 0.889743 0.080011 -0.420318 -0.403144 1.126439 -0.361815 0.929077 -0.713509 0.985332 0.242538 -0.917072 -0.057393 -0.488874 -0.723317 0.060555 -0.678093 -0.134172 0.077537 0.729663 -0.295301 -0.213575 -0.516139 -0.449349 -0.487124 0.033850 -0.376786 0.270049 0.885131 0.300100 1.139350 -1.038440 -1.193752 -0.475536 -0.798919 0.556193 -0.958690 -0.814718 -0.508433 -0.014669 1.064134 0.082489 1.132092 0.215406 0.098833 0.954594 -0.168210 -0.174761 -0.348583 -0.372312 -0.493933 -0.133547 0.962655 -0.184195 -0.474825 -0.048982 -0.916607 -0.292659 -0.150419 0.360300 -0.660115 -0.661864
            // -1.153229 1.115101 0.451486 -0.019335 0.081046 -0.897013 0.066294 -1.108879 0.481739 0.346830 0.132454 -0.600972 -0.239763 -0.759759 0.030469 -0.030816 0.559743 -1.076532 0.109714 1.108368 0.673998 0.905330 -0.905321 0.225631 -0.697544 0.915237 1.162965 1.048784 0.407300 -0.826737 -0.931706 0.486326 0.330038 -0.044726 -1.078226 0.321167 -0.604617 -0.481830 -0.058719 -0.136195 0.525000 0.013513 1.152715 0.996482 -1.052405 0.911974 0.466837 1.038911 -0.832899 -1.179938 -0.139061 -0.652416 -0.069374 0.994420 -0.600082 -0.625005 -1.008501 0.855489 0.648194 0.321766 1.183240 -1.127553 0.181576 -0.139220 0.062323 -0.616375 0.171741 0.632307 0.652488 -0.686222 0.935554 0.007636
            // 0.645204 -0.420033 -0.537861 -0.392927 0.903564 1.115874 0.434954 -0.534552 -0.133268 1.017383 -0.683493 -1.091488 -0.736239 -0.041714 0.358751 -0.685861 -0.769060 1.112335 -0.141346 -0.952757 0.229603 -1.131032 -1.030214 -0.773239 -0.956746 0.503800 -0.769620 0.473041 0.161575 0.728127 0.029532 0.009683 0.155976 1.116576 0.142849 -1.017897 1.063033 0.776606 0.393974 -0.655821 1.118171 0.578225 0.157362 0.364287 -1.085704 -0.129203 -0.303116 -0.940305 0.807298 0.912751 1.012666 0.357233 0.498155 -1.145789 -0.845446 -0.820021 0.392601 -0.333279 0.326482 0.283910 -0.719092 0.220219 1.008839 -0.640068 -1.047775 0.687632 0.976318 0.989751 0.060505 -1.020957 -0.990341 0.228239
            // -0.057175 0.855893 -0.222570 0.413212 0.001550 -0.518960 -0.092583 -1.101160 0.064024 -0.143040 -0.696054 0.337081 -0.645138 0.285723 -1.035636 -0.911924 0.622264 -0.016157 0.793384 -0.151767 -0.398511 -1.093932 1.106506 -0.048716 0.153941 -1.182110 0.858962 -0.271042 0.243307 -0.137128 0.532404 0.962164 -0.498810 -0.680566 -0.202586 -0.152808 -0.250392 1.110015 0.237943 0.554633 -0.933662 0.495773 -0.334821 0.599023 -0.382504 0.970424 0.159727 -0.806215 0.405711 -0.806857 0.602087 0.708214 0.710844 0.224632 -0.297654 -0.906908 0.413975 0.748755 1.085093 0.402713 0.654039 -0.237107 0.815357 0.011949 0.108247 1.058832 -0.610000 -0.945535 0.651756 -0.850273 -1.037817 0.250334
            // 0.627970 0.601065 0.232290 1.116468 -0.487375 -1.140154 0.753311 0.381261 -0.131786 -0.109594 0.616738 0.519294 -0.399424 -0.541150 -0.207198 -0.019238 0.491733 0.381305 0.611392 1.048191 0.235917 0.888293 0.597352 0.190067 -0.305590 -1.002172 1.101423 -1.116615 -0.296031 0.402004 0.604487 0.491015 -0.774655 0.524382 -0.137080 -0.982569 -0.636777 0.248829 -0.601824 -1.159025 0.113131 0.734116 -0.019931 0.073767 -0.235716 -0.756500 -0.323380 0.105316 0.727886 0.931194 -0.789581 -0.939996 -0.273909 0.043045 0.822099 -0.914377 0.056735 0.243902 -1.004455 -0.421837 0.958358 -0.150269 -0.380719 -0.872485 0.117334 -0.428980 -0.058306 -0.443280 0.282252 0.457911 -0.520553 0.187160
            // 0.445298 -0.365835 -0.979644 0.166970 0.342245 0.574933 0.597103 -0.941189 0.363747 0.942836 1.039003 0.328763 -0.296133 -0.297388 1.133498 -0.212284 0.630429 -1.052410 0.390220 0.243479 -1.088395 0.840149 0.956629 -0.465953 -0.334023 -0.878854 0.436358 1.182090 0.162003 1.191590 -0.056629 0.842802 -0.860613 -0.153127 -0.904862 -0.822966 -0.338993 1.137824 0.780096 0.101463 -0.300451 -0.882993 0.869597 -0.528300 0.781236 0.354585 1.101279 -0.833491 -0.737721 1.116310 0.207504 0.081174 -0.568636 -0.138814 -1.054337 -0.752827 -1.144685 -0.088285 0.995978 -0.262308 0.882332 0.087306 0.346484 -0.990146 -0.778511 0.706102 0.692255 -1.163704 -1.190496 0.521891 -0.002426 -0.901865
            // -1.147564 0.905074 -1.018686 0.173556 0.237254 0.375398 -0.104201 -1.114410 0.586780 -0.405159 -0.380734 1.017159 -0.018111 0.163401 0.055064 -0.582203 0.349598 -0.505840 0.077280 0.533594 0.742503 -0.251496 -0.476883 0.475508 0.883403 -0.413209 -0.591443 -0.460440 0.862571 -0.887767 -0.982322 -1.175616 -0.139273 1.168406 0.734907 -0.246711 -1.032493 -0.906548 0.034380 -0.525988 0.839772 0.332087 1.180328 -0.876850 1.059176 0.737319 -0.934478 0.245452 0.255574 -0.472154 -0.019407 0.488610 1.048407 -0.834877 -0.708560 -0.456748 -0.205030 -0.950764 1.026063 1.126783 -1.159270 -1.089354 0.932474 -0.856920 0.414548 0.098367 -0.110362 0.774629 -0.446793 -1.105836 0.684619 -0.985245
            // ------------------------
            // bottom:
            // shape: 9 x 7 x (1 pack 1)
            // 67.000000 -98.000000 86.000000 83.000000 94.000000 -73.000000 -126.000000 17.000000 0.000000
            // 51.000000 117.000000 95.000000 9.000000 -45.000000 -43.000000 120.000000 -38.000000 99.000000
            // -76.000000 105.000000 26.000000 -98.000000 -6.000000 -52.000000 -77.000000 6.000000 -72.000000
            // -14.000000 8.000000 78.000000 -31.000000 -23.000000 -55.000000 -48.000000 -52.000000 4.000000
            // -40.000000 29.000000 94.000000 32.000000 121.000000 -110.000000 -127.000000 -51.000000 -85.000000
            // 59.000000 -102.000000 -87.000000 -54.000000 -2.000000 113.000000 9.000000 120.000000 23.000000
            // 11.000000 102.000000 -18.000000 -19.000000 -37.000000 -40.000000 -53.000000 -14.000000 102.000000
            // ------------------------
        }

        if (bottom_blob_int8.empty())
            return -100;
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob_int8, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    int w = bottom_blob_bordered.w;
    int h = bottom_blob_bordered.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob_bordered.elempack;

    fprintf(stderr, "bottom: \n");
    pretty_print<const signed char>(bottom_blob_int8);
    fprintf(stderr, "bottom bordered: \n");
    pretty_print<const signed char>(bottom_blob_bordered);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    bool use_int8_requantize = int8_scale_term > 100;
    int out_elempack = 1;
    int out_elempack_int32 = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        if (use_int8_requantize)
        {
            out_elempack = num_output % packn == 0 ? packn : 1;
            out_elempack_int32 = num_output % packn == 0 ? packn : 1;
        }
        else
        {
            out_elempack = num_output % packn_32 == 0 ? packn_32 : 1;
            out_elempack_int32 = num_output % packn_32 == 0 ? packn_32 : 1;
        }
    }
#endif // __riscv_vector
    size_t out_elemsize = use_int8_requantize ? 1u * out_elempack : 4u * out_elempack;
#if NCNN_ZFH
    if (support_fp16_storage && opt.use_fp16_storage)
    {
        out_elemsize = use_int8_requantize ? 1u * out_elempack : 2u * out_elempack;
    }
#endif // NCNN_ZFH
    if (opt.use_bf16_storage)
        out_elemsize = use_int8_requantize ? 1u * out_elempack : 2u * out_elempack;

    Mat top_blob_int32;
    top_blob_int32.create(outw, outh, num_output / out_elempack_int32, (size_t)(4u * out_elempack_int32), out_elempack_int32, opt.workspace_allocator);
    if (top_blob_int32.empty())
        return -100;

    // TODO: Implement winograd, sgemm, etc
    convolution_packed_int8_rvv(bottom_blob_bordered, top_blob_int32, weight_data_tm, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
    bottom_blob_bordered.release();

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (use_int8_requantize)
    {
        requantize_from_int32_to_int8(top_blob_int32, top_blob, scale_in_data, top_blob_int8_scales, bias_data, activation_type, activation_params, opt);
    }
    else
    {
        dequantize_from_int32(top_blob_int32, top_blob, scale_in_data, bias_data, opt);

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }
    }
    return 0;
}
#endif // NCNN_INT8

} // namespace ncnn
