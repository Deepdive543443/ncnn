#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>

// NCNN headers
#include "net.h"
#include "layer.h"

// Helper function to print a 3D ncnn::Mat
void print_mat(const ncnn::Mat& m, const char* name)
{
    std::cout << "=== " << name << " (w=" << m.w << ", h=" << m.h << ", c=" << m.c << ") ===\n";
    for (int q = 0; q < m.c; q++)
    {
        std::cout << "Channel " << q << ":\n";
        const float* ptr = m.channel(q);
        for (int y = 0; y < m.h; y++)
        {
            for (int x = 0; x < m.w; x++)
            {
                // Print with fixed precision for alignment
                std::cout << std::setw(5) << std::fixed << std::setprecision(1) << ptr[x] << " ";
            }
            std::cout << "\n";
            ptr += m.w; // advance pointer to the next row
        }
    }
    std::cout << "====================================\n\n";
}

int main(int argc, char** argv)
{
    // Seed the random number generator
    std::srand(std::time(nullptr));

    // 1. Create a 3D input matrix (width=3, height=3, channels=4)
    int w, h, c;
    if (argc > 3)
    {
        w = std::atoi(argv[1]);
        h = std::atoi(argv[2]);
        c = std::atoi(argv[3]);
    }
    else
    {
        w = 3;
        h = 3;
        c = 4;
    }

    ncnn::Mat in(w, h, c);

    // Fill with random float values (e.g., 0.0 to 9.9)
    for (int q = 0; q < c; q++)
    {
        float* ptr = in.channel(q);
        for (int i = 0; i < w * h; i++)
        {
            ptr[i] = static_cast<float>(std::rand() % 100) / 10.0f;

            // TIP: If you want to visually verify the shuffle easily,
            // uncomment the line below to fill each channel with its index number instead:
            // ptr[i] = static_cast<float>(q);
        }
    }

    print_mat(in, "Input Matrix");

    // 2. Create the ShuffleChannel layer directly
    ncnn::Layer* shuffle = ncnn::create_layer("ShuffleChannel");

    // 3. Configure the layer parameters
    // In NCNN, parameter index 0 for ShuffleChannel represents the 'group' count.
    ncnn::ParamDict pd;
    int groups = 2; // Split the 4 channels into 2 groups before shuffling
    pd.set(0, groups);
    shuffle->load_param(pd);

    // 4. Run the forward pass
    ncnn::Mat out;
    ncnn::Option opt;

    // Disable packing and fp16 storage to guarantee standard float32 output
    // This prevents segmentation faults when reading the pointers directly.
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    shuffle->forward(in, out, opt);

    // 5. Print the output
    print_mat(out, "Output Matrix (After ShuffleChannel)");

    // 6. Cleanup memory allocated by create_layer
    delete shuffle;

    return 0;
}