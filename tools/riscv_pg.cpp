#include <iostream>
#include <vector>
#include <stdlib.h>
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
    const size_t vl = csrr_vlenb();
    const size_t vlmax_e32m8 = __riscv_vsetvlmax_e32m8();
    const size_t vlmax_e16m8 = __riscv_vsetvlmax_e16m8();
    const size_t vlmax_e8m8 = __riscv_vsetvlmax_e8m8();

    std::cout << "Vector Length in bytes: " << vl << std::endl;
    std::cout << "Maximum 32 bit elements: " << vlmax_e32m8 << std::endl;
    std::cout << "Maximum 16 bit elements: " << vlmax_e16m8 << std::endl;
    std::cout << "Maximum  8 bit elements: " << vlmax_e8m8 << std::endl;

    std::vector<int32_t> s(vlmax_e32m8, 0);

    vint8m2_t vs = __riscv_vmv_v_x_i8m2(2, vlmax_e32m8);
    vint8m2_t va = __riscv_vmv_v_x_i8m2(4, vlmax_e32m8);
    vint16m4_t vb = __riscv_vmv_v_x_i16m4(1, vlmax_e32m8);
    
    vint32m8_t vresult = __riscv_vwadd_vv_i32m8(vb, __riscv_vwmul_vv_i16m4(vs, va, vlmax_e32m8), vlmax_e32m8);

    __riscv_vse32_v_i32m8(&s[0], vresult, vlmax_e32m8);

    std::cout << "[ ";
    for (int32_t &rs : s) {
        std::cout << rs << " ";
    }
    std::cout << "]" << std::endl;
    

    return 0;
#else
    std::cout << "RISC-V Vector extension not supported" << std::endl;
    return 1;
#endif
}