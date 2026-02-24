#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include "mat.h"

#define W            2
#define H            2
#define D            3
#define CHANNEL_SIZE W* H* D
#define C            16
#define PACK_SIZE    16

int main(int argc, char** argv)
{
    ncnn::Mat unpacked, packed;
    unpacked.create(W, H, D, C, (size_t)1u, 1);
    {
        uint8_t idx = 0;
        for (size_t c = 0; c < C; c++)
        {
            uint8_t* ptr = unpacked.channel(c);
            for (size_t j = 0; j < CHANNEL_SIZE; j++)
            {
                ptr[j] = ++idx;
            }
        }
    }

    ncnn::convert_packing(unpacked, packed, PACK_SIZE);

    std::cout << "Unpacked layout:" << unpacked.elemsize << " bytes " << unpacked.elempack << " pack" << std::endl;
    std::cout << "w: " << unpacked.w << std::endl;
    std::cout << "h: " << unpacked.h << std::endl;
    std::cout << "d: " << unpacked.d << std::endl;
    std::cout << "c: " << unpacked.c << std::endl;
    std::cout << "cstep: " << unpacked.cstep << std::endl;
    std::cout << "elemsize: " << unpacked.elemsize << std::endl;
    std::cout << "total: " << unpacked.total() << std::endl;
    std::cout << "flat bytes: " << std::endl;
    uint8_t* data_ptr = (uint8_t*)unpacked.data;
    for (size_t i = 0; i < unpacked.total() * unpacked.elemsize; i++)
    {
        std::cout << (int)data_ptr[i] << " ";
        if ((i + 1) % PACK_SIZE == 0)
        {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    std::cout << "Packed layout:" << packed.elemsize << " bytes " << packed.elempack << " pack" << std::endl;
    std::cout << "w: " << packed.w << std::endl;
    std::cout << "h: " << packed.h << std::endl;
    std::cout << "d: " << packed.d << std::endl;
    std::cout << "c: " << packed.c << std::endl;
    std::cout << "cstep: " << packed.cstep << std::endl;
    std::cout << "elemsize: " << packed.elemsize << std::endl;
    std::cout << "total: " << packed.total() << std::endl;

    data_ptr = (uint8_t*)packed.data;
    for (size_t i = 0; i < packed.total() * packed.elemsize; i++)
    {
        std::cout << (int)data_ptr[i] << " ";
        if ((i + 1) % PACK_SIZE == 0)
        {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    return 0;
}