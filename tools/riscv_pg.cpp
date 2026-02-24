#include <iostream>
#if __riscv_vector
#include <riscv_vector.h>

static inline int csrr_vlenb()
{
    int a = 0;
    asm volatile("csrr %0, vlenb"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}
#endif

int main(int argc, char** argv)
{
#if __riscv_vector
    size_t vl = csrr_vlenb();
    std::cout << "VLENB: " << vl << std::endl;
    std::cout << "VLEN BIT: " << vl * 8 << std::endl;
    std::cout << "e8x" << vl / 2 << std::endl;
    std::cout << "e16x" << vl / 4 << std::endl;
    std::cout << "e32x" << vl / 8 << std::endl;
    return 0;
#else
    std::cout << "RISC-V Vector extension not supported" << std::endl;
    return 1;
#endif
}