// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Minimal demo that runs the ShuffleChannel layer on a deterministic input
// with the packed layout enabled, then prints the result.
//
// Purpose: easy before/after comparison for the RISC-V ShuffleChannel packed
// _group == 2 fix. Because the input is fixed and the computation is purely
// integer-like (we feed sequential floats), any difference in the output
// between a buggy and a correct build is immediately visible.
//
// Expected correct behavior (for group=2, reverse=0, channels=C, H*W=N):
//   input:  channel c contains the values [c*N .. c*N + N - 1]
//   output: channel (2*j + i) must equal input channel (channels/2 * i + j),
//           i.e. output[0] == input[0], output[1] == input[C/2],
//                output[2] == input[1], output[3] == input[C/2 + 1], ...
//
// Usage:
//   shufflechannel_demo [w] [h] [channels] [group] [reverse]
//
// Defaults are chosen to hit the packed _group == 2 && channels % 2 == 0
// path on typical RVV targets (packn=4 for fp32).

#include "layer.h"
#include "mat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_sequential(ncnn::Mat& m)
{
    const int size = m.w * m.h;
    for (int c = 0; c < m.c; c++)
    {
        float* p = m.channel(c);
        for (int i = 0; i < size; i++)
        {
            p[i] = (float)(c * size + i);
        }
    }
}

static void print_mat(const char* tag, const ncnn::Mat& m)
{
    fprintf(stdout, "%s: w=%d h=%d c=%d elempack=%d\n",
            tag, m.w, m.h, m.c, m.elempack);

    // Always print in pack1 view so the output is independent of the packed
    // storage layout and directly comparable across builds/targets.
    ncnn::Mat m1;
    if (m.elempack != 1)
    {
        ncnn::Option opt;
        opt.use_packing_layout = true;
        ncnn::convert_packing(m, m1, 1, opt);
    }
    else
    {
        m1 = m;
    }

    const int size = m1.w * m1.h;
    for (int c = 0; c < m1.c; c++)
    {
        const float* p = m1.channel(c);
        fprintf(stdout, "  ch%02d:", c);
        for (int i = 0; i < size; i++)
        {
            fprintf(stdout, " %6.0f", p[i]);
        }
        fprintf(stdout, "\n");
    }
}

int main(int argc, char** argv)
{
    int w = 5;
    int h = 7;
    int channels = 16;
    int group = 2;
    int reverse = 0;

    if (argc >= 2) w = atoi(argv[1]);
    if (argc >= 3) h = atoi(argv[2]);
    if (argc >= 4) channels = atoi(argv[3]);
    if (argc >= 5) group = atoi(argv[4]);
    if (argc >= 6) reverse = atoi(argv[5]);

    fprintf(stdout, "ShuffleChannel demo: w=%d h=%d channels=%d group=%d reverse=%d\n",
            w, h, channels, group, reverse);

    // 1) build a pack1 input filled with sequential floats
    ncnn::Mat in_pack1(w, h, channels);
    fill_sequential(in_pack1);

    // 2) force packed layout (this is the path the RISC-V bug lives in)
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_bf16_storage = false;

    ncnn::Mat in_packed;
    ncnn::convert_packing(in_pack1, in_packed, 0 /* auto pick max */, opt);

    print_mat("input (pack1 view)", in_packed);

    // 3) create and configure the ShuffleChannel layer
    ncnn::Layer* op = ncnn::create_layer("ShuffleChannel");
    if (op == 0)
    {
        fprintf(stderr, "create_layer(ShuffleChannel) failed\n");
        return -1;
    }

    ncnn::ParamDict pd;
    pd.set(0, group);
    pd.set(1, reverse);
    op->load_param(pd);

    op->create_pipeline(opt);

    // 4) forward
    ncnn::Mat out_packed;
    int ret = op->forward(in_packed, out_packed, opt);
    if (ret != 0)
    {
        fprintf(stderr, "ShuffleChannel::forward failed ret=%d\n", ret);
        op->destroy_pipeline(opt);
        delete op;
        return -1;
    }

    print_mat("output (pack1 view)", out_packed);

    // 5) verify against the naive reference
    ncnn::Mat out_unpacked;
    if (out_packed.elempack != 1)
    {
        convert_packing(out_packed, out_unpacked, 1, opt);
    }
    else
    {
        out_unpacked = out_packed;
    }

    const int size = w * h;
    const int _group = reverse ? channels / group : group;
    const int channels_per_group = channels / _group;

    int mismatches = 0;
    for (int i = 0; i < _group; i++)
    {
        for (int j = 0; j < channels_per_group; j++)
        {
            const int src_q = channels_per_group * i + j;
            const int dst_q = _group * j + i;
            const float* p_out = out_unpacked.channel(dst_q);
            const float base = (float)(src_q * size);
            for (int k = 0; k < size; k++)
            {
                const float expected = base + (float)k;
                if (p_out[k] != expected)
                {
                    if (mismatches < 16)
                    {
                        fprintf(stdout,
                                "MISMATCH ch=%d idx=%d expected=%.0f got=%.0f\n",
                                dst_q, k, expected, p_out[k]);
                    }
                    mismatches++;
                }
            }
        }
    }

    if (mismatches == 0)
    {
        fprintf(stdout, "OK: output matches naive reference\n");
    }
    else
    {
        fprintf(stdout, "FAIL: %d mismatches against naive reference\n", mismatches);
    }

    op->destroy_pipeline(opt);
    delete op;

    return mismatches == 0 ? 0 : 1;
}
