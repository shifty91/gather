/**
 * Copyright (C) 2015 Kurt Kanzenbach <kurt@kmk-computers.de>
 *
 * This program collects some ideas of how to implement gather
 * and scatter operations using vectorization (SSE/AVX/AVX512).
 * Datatypes: Double, Float, Int32
 *
 * Runtime is measured in CPU cycles.
 *
 */

#include <iostream>
#include <string>
#include <cstring>
#include <cstdint>

// include correct header
#ifdef __AVX__
#include <immintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

/**
 * Macros for disabling specific warnings.
 */
#if defined(__GNUC__) && !defined(__clang__)
#define DISABLE_WARNING_UNINITIALIZED             \
    _Pragma("GCC diagnostic push")                         \
    _Pragma("GCC diagnostic ignored \"-Wuninitialized\"")
#define ENABLE_WARNING_UNINITIALIZED   \
    _Pragma("GCC diagnostic pop")
#endif

#ifdef __clang__
#define DISABLE_WARNING_UNINITIALIZED              \
    _Pragma("clang diagnostic push")                        \
    _Pragma("clang diagnostic ignored \"-Wuninitialized\"")
#define ENABLE_WARNING_UNINITIALIZED   \
    _Pragma("clang diagnostic pop")
#endif

#ifndef DISABLE_WARNING_UNINITIALIZED
#define DISABLE_WARNING_UNINITIALIZED
#endif
#ifndef ENABLE_WARNING_UNINITIALIZED
#define ENABLE_WARNING_UNINITIALIZED
#endif

/**
 * This measurement idea comes from:
 * http://www.intel.com/content/dam/www/public/us/en/documents/white-papers/ia-32-ia-64-benchmark-code-execution-paper.pdf
 */
#ifdef WITH_PERF
#define PERF_START                                                      \
    std::uint64_t t1, t2;                                               \
    std::int32_t low, high;                                             \
    asm volatile ("cpuid\n\t"                                           \
                  "rdtsc\n\t"                                           \
                  "mov %%edx, %0\n\t"                                   \
                  "mov %%eax, %1\n\t": "=r" (high), "=r" (low)::        \
                  "%rax", "%rbx", "%rcx", "%rdx");                      \
    t1 = static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32);
#else
#define PERF_START
#endif

#ifdef WITH_PERF
#define PERF_END                                                        \
    asm volatile("rdtscp\n\t"                                           \
                 "mov %%edx, %0\n\t"                                    \
                 "mov %%eax, %1\n\t"                                    \
                 "cpuid\n\t": "=r" (high), "=r" (low)::                 \
                 "%rax", "%rbx", "%rcx", "%rdx");                       \
    t2 = static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32); \
    std::cout << "Perf: " << __func__ << ": " << (t2 - t1) << std::endl;
#else
#define PERF_END
#endif

template<typename TYPE> static
void print_array(const TYPE *ptr, unsigned len)
{
    std::cout << "[";
    for (unsigned i = 0; i < len; ++i) {
        std::cout << ptr[i];
        if (i != (len - 1))
            std::cout << ",";
    }
    std::cout << "]" << std::endl;
}

#define SCATTER_FLOAT(value, method)                        \
    do {                                                    \
        std::memset(fscatter, '\0', sizeof(float) * LEN);   \
        method((value), fscatter, idx);                     \
        print_array(fscatter, LEN);                         \
    } while (0)

#define SCATTER_DOUBLE(value, method)                       \
    do {                                                    \
        std::memset(dscatter, '\0', sizeof(double) * LEN);  \
        method((value), dscatter, idx);                     \
        print_array(dscatter, LEN);                         \
    } while (0)

#define SCATTER_INT32(value, method)                                \
    do {                                                            \
        std::memset(iscatter, '\0', sizeof(std::int32_t) * LEN);    \
        method((value), iscatter, idx);                             \
        print_array(iscatter, LEN);                                 \
    } while (0)

#ifdef __SSE2__
// print __m** data types
static
void print_vec_sse(const __m128& vec)
{
    const float *ptr = reinterpret_cast<const float *>(&vec);
    std::cout << "[" << *ptr << ", " << *(ptr + 1) << ", "
              << *(ptr + 2) << ", " << *(ptr + 3) << "]" << std::endl;
}

static
void print_vec_sse(const __m128d& vec)
{
    const double *ptr = reinterpret_cast<const double *>(&vec);
    std::cout << "[" << *ptr << ", " << *(ptr + 1)
              << "]" << std::endl;
}

static
void print_vec_sse(const __m128i& vec)
{
    const int *ptr = reinterpret_cast<const int *>(&vec);
    std::cout << "[" << *ptr << ", " << *(ptr + 1) << ", "
              << *(ptr + 2) << ", " << *(ptr + 3) << "]" << std::endl;
}
#endif

#ifdef __AVX__
static
void print_vec_avx(const __m256& vec)
{
    const float *ptr = reinterpret_cast<const float *>(&vec);
    std::cout << "[" << *ptr << ", " << *(ptr + 1) << ", "
              << *(ptr + 2) << ", " << *(ptr + 3) << ", "
              << *(ptr + 4) << ", " << *(ptr + 5) << ", "
              << *(ptr + 6) << ", " << *(ptr + 7)
              << "]" << std::endl;
}

static
void print_vec_avx(const __m256d& vec)
{
    const double *ptr = reinterpret_cast<const double *>(&vec);
    std::cout << "[" << *ptr << ", " << *(ptr + 1) << ", "
              << *(ptr + 2) << ", " << *(ptr + 3)
              << "]" << std::endl;
}
#endif

#ifdef __AVX512F__
static
void print_vec_avx512(const __m512& vec)
{
    const float *ptr = reinterpret_cast<const float *>(&vec);
    std::cout << "[" << *ptr << ", " << *(ptr + 1) << ", "
              << *(ptr + 2) << ", " << *(ptr + 3) << ", "
              << *(ptr + 4) << ", " << *(ptr + 5) << ", "
              << *(ptr + 6) << ", " << *(ptr + 7) << ", "
              << *(ptr + 8) << ", " << *(ptr + 9) << ", "
              << *(ptr + 10) << ", " << *(ptr + 11) << ", "
              << *(ptr + 12) << ", " << *(ptr + 13) << ", "
              << *(ptr + 14) << ", " << *(ptr + 15)
              << "]" << std::endl;
}

static
void print_vec_avx512(const __m512d& vec)
{
    const double *ptr = reinterpret_cast<const double *>(&vec);
    std::cout << "[" << *ptr << ", " << *(ptr + 1) << ", "
              << *(ptr + 2) << ", " << *(ptr + 3) << ", "
              << *(ptr + 4) << ", " << *(ptr + 5) << ", "
              << *(ptr + 6) << ", " << *(ptr + 7)
              << "]" << std::endl;
}
#endif

// AVX512F section
#ifdef __AVX512F__
__m512d __attribute__((noinline))
gather_avx512_double_gather(const double *ptr, const unsigned *offsets)
{
    __m512d result;
    __m256i indices;

    PERF_START;
    indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets));
    result  = _mm512_i32gather_pd(indices, ptr, 8);
    PERF_END;

    return result;
}

__m512 __attribute__((noinline))
gather_avx512_float_gather(const float *ptr, const unsigned *offsets)
{
    __m512 result;
    __m512i indices;

    PERF_START;
    indices = _mm512_load_epi32(offsets);
    result  = _mm512_i32gather_ps(indices, ptr, 4);
    PERF_END;

    return result;
}

void __attribute__((noinline))
scatter_avx512_double_scatter(const __m512d& value, double *ptr, const unsigned *offsets)
{
    __m256i indices;

    PERF_START;
    indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets));
    _mm512_i32scatter_pd(ptr, indices, value, 8);
    PERF_END;
}

void __attribute__((noinline))
scatter_avx512_float_scatter(const __m512& value, float *ptr, const unsigned *offsets)
{
    __m512i indices;

    PERF_START;
    indices = _mm512_load_epi32(offsets);
    _mm512_i32scatter_ps(ptr, indices, value, 4);
    PERF_END;
}
#endif

// AVX2 section
#ifdef __AVX2__
__m256d __attribute__((noinline))
gather_avx2_double_gather(const double *ptr, const unsigned *offsets)
{
    __m256d result;
    __m128i indices;

    PERF_START;
    indices = _mm_loadu_si128(reinterpret_cast<const __m128i *>(offsets));
    result  = _mm256_i32gather_pd(ptr, indices, 8);
    PERF_END;

    return result;
}

__m256 __attribute__((noinline))
gather_avx2_float_gather(const float *ptr, const unsigned *offsets)
{
    __m256 result;
    __m256i indices;

    PERF_START;
    indices = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(offsets));
    result  = _mm256_i32gather_ps(ptr, indices, 4);
    PERF_END;

    return result;
}
#endif

// AVX section
#ifdef __AVX__
DISABLE_WARNING_UNINITIALIZED
__m256d __attribute__((noinline))
gather_avx_double_insert(const double *ptr, const unsigned *offsets)
{
    __m256d result;
    __m128d tmp;

    PERF_START;
    tmp     = _mm_loadl_pd(tmp, ptr + offsets[0]);
    tmp     = _mm_loadh_pd(tmp, ptr + offsets[1]);
    result  = _mm256_insertf128_pd(result, tmp, 0);
    tmp     = _mm_loadl_pd(tmp, ptr + offsets[2]);
    tmp     = _mm_loadh_pd(tmp, ptr + offsets[3]);
    result  = _mm256_insertf128_pd(result, tmp, 1);
    PERF_END;

    return result;
}
ENABLE_WARNING_UNINITIALIZED

__m256d __attribute__((noinline))
gather_avx_double_set(const double *ptr, const unsigned *offsets)
{
    __m256d result;

    PERF_START;
    result = _mm256_set_pd(ptr[offsets[3]], ptr[offsets[2]],
                           ptr[offsets[1]], ptr[offsets[0]]);
    PERF_END;

    return result;
}

__m256d __attribute__((noinline))
gather_avx_double_load(const double *ptr, const unsigned *offsets)
{
    __m256d result;

    PERF_START;
    double tmp[4] __attribute__((aligned (32))) = { ptr[offsets[0]], ptr[offsets[1]],
                                                    ptr[offsets[2]], ptr[offsets[3]] };
    result = _mm256_load_pd(tmp);
    PERF_END;

    return result;
}

void __attribute__((noinline))
scatter_avx_double_extract(const __m256d& value, double *ptr, const unsigned *offsets)
{
    __m128d tmp;

    PERF_START;
    tmp = _mm256_extractf128_pd(value, 0);
    _mm_storel_pd(ptr + offsets[0], tmp);
    _mm_storeh_pd(ptr + offsets[1], tmp);
    tmp = _mm256_extractf128_pd(value, 1);
    _mm_storel_pd(ptr + offsets[2], tmp);
    _mm_storeh_pd(ptr + offsets[3], tmp);
    PERF_END;
}

void __attribute__((noinline))
scatter_avx_double_cast(const __m256d& value, double *ptr, const unsigned *offsets)
{
    PERF_START;
    const double *p = reinterpret_cast<const double *>(&value);
    ptr[offsets[0]] = p[0];
    ptr[offsets[1]] = p[1];
    ptr[offsets[2]] = p[2];
    ptr[offsets[3]] = p[3];
    PERF_END;
}

void __attribute__((noinline))
scatter_avx_double_store(const __m256d& value, double *ptr, const unsigned *offsets)
{
    PERF_START;
    double tmp[4] __attribute__((aligned (32)));
    _mm256_store_pd(tmp, value);
    ptr[offsets[0]] = tmp[0];
    ptr[offsets[1]] = tmp[1];
    ptr[offsets[2]] = tmp[2];
    ptr[offsets[3]] = tmp[3];
    PERF_END;
}

DISABLE_WARNING_UNINITIALIZED
__m256 __attribute__((noinline))
gather_avx_float_insert(const float *ptr, const unsigned *offsets)
{
    __m256 result;
    __m128 tmp;

    PERF_START;
    tmp    = _mm_load_ss(ptr + offsets[0]);
    tmp    = _mm_insert_ps(tmp, _mm_load_ss(ptr + offsets[1]), _MM_MK_INSERTPS_NDX(0,1,0));
    tmp    = _mm_insert_ps(tmp, _mm_load_ss(ptr + offsets[2]), _MM_MK_INSERTPS_NDX(0,2,0));
    tmp    = _mm_insert_ps(tmp, _mm_load_ss(ptr + offsets[3]), _MM_MK_INSERTPS_NDX(0,3,0));
    result = _mm256_insertf128_ps(result, tmp, 0);
    tmp    = _mm_load_ss(ptr + offsets[4]);
    tmp    = _mm_insert_ps(tmp, _mm_load_ss(ptr + offsets[5]), _MM_MK_INSERTPS_NDX(0,1,0));
    tmp    = _mm_insert_ps(tmp, _mm_load_ss(ptr + offsets[6]), _MM_MK_INSERTPS_NDX(0,2,0));
    tmp    = _mm_insert_ps(tmp, _mm_load_ss(ptr + offsets[7]), _MM_MK_INSERTPS_NDX(0,3,0));
    result = _mm256_insertf128_ps(result, tmp, 1);
    PERF_END;

    return result;
}
ENABLE_WARNING_UNINITIALIZED

/**
 * Same as above just for AVX.
 * Instruction: vinsertps xmm, xmm, xmm/m32, imm8
 *
 * @param a xmm register
 * @param base base pointer
 * @param offset offset
 * @param idx index, has to be a constant number like 0x10, no variable
 */
#define SHORTVEC_INSERT_PS_AVX(a, base, offset, idx)                    \
    do {                                                                \
        asm volatile ("vinsertps %1, (%q2, %q3, 4), %0, %0\n"           \
                      : "+x" (a) : "N" (idx), "r" (base), "r" (offset) : "memory"); \
    } while (0)

DISABLE_WARNING_UNINITIALIZED
__m256 __attribute__((noinline))
gather_avx_float_insert2(const float *ptr, const unsigned *offsets)
{
    __m256 result;
    __m128 tmp;

    PERF_START;
    tmp    = _mm_load_ss(ptr + offsets[0]);
    SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[1], _MM_MK_INSERTPS_NDX(0,1,0));
    SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[2], _MM_MK_INSERTPS_NDX(0,2,0));
    SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[3], _MM_MK_INSERTPS_NDX(0,3,0));
    result = _mm256_insertf128_ps(result, tmp, 0);
    tmp    = _mm_load_ss(ptr + offsets[4]);
    SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[5], _MM_MK_INSERTPS_NDX(0,1,0));
    SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[6], _MM_MK_INSERTPS_NDX(0,2,0));
    SHORTVEC_INSERT_PS_AVX(tmp, ptr, offsets[7], _MM_MK_INSERTPS_NDX(0,3,0));
    result = _mm256_insertf128_ps(result, tmp, 1);
    PERF_END;

    return result;
}

__m256 __attribute__((noinline))
gather_avx_float_insert3(const float *ptr, const unsigned *offsets)
{
    __m256 result;
    __m128 tmp1, tmp2;

    PERF_START;
    tmp1   = _mm_load_ss(ptr + offsets[0]);
    SHORTVEC_INSERT_PS_AVX(tmp1, ptr, offsets[1], _MM_MK_INSERTPS_NDX(0,1,0));
    tmp2   = _mm_load_ss(ptr + offsets[2]);
    SHORTVEC_INSERT_PS_AVX(tmp2, ptr, offsets[3], _MM_MK_INSERTPS_NDX(0,1,0));
    tmp1   = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
    result = _mm256_insertf128_ps(result, tmp1, 0);
    tmp1   = _mm_load_ss(ptr + offsets[4]);
    SHORTVEC_INSERT_PS_AVX(tmp1, ptr, offsets[5], _MM_MK_INSERTPS_NDX(0,1,0));
    tmp2   = _mm_load_ss(ptr + offsets[6]);
    SHORTVEC_INSERT_PS_AVX(tmp2, ptr, offsets[7], _MM_MK_INSERTPS_NDX(0,1,0));
    tmp1   = _mm_shuffle_ps(tmp1, tmp2, _MM_SHUFFLE(1,0,1,0));
    result = _mm256_insertf128_ps(result, tmp1, 1);
    PERF_END;

    return result;
}

__m256 __attribute__((noinline))
gather_avx_float_insert4(const float *ptr, const unsigned *offsets)
{
    __m256 result;
    __m128 tmp;

    PERF_START;
#if !defined(__clang__) || (__clang_major__ >= 3 && __clang_minor__ >= 5)
    tmp    = _mm_load_ss(ptr + offsets[0]);
    tmp    = _mm_insert_ps(tmp, *reinterpret_cast<const __m128 *>(ptr + offsets[1]), _MM_MK_INSERTPS_NDX(0,1,0));
    tmp    = _mm_insert_ps(tmp, *reinterpret_cast<const __m128 *>(ptr + offsets[2]), _MM_MK_INSERTPS_NDX(0,2,0));
    tmp    = _mm_insert_ps(tmp, *reinterpret_cast<const __m128 *>(ptr + offsets[3]), _MM_MK_INSERTPS_NDX(0,3,0));
    result = _mm256_insertf128_ps(result, tmp, 0);
    tmp    = _mm_load_ss(ptr + offsets[4]);
    tmp    = _mm_insert_ps(tmp, *reinterpret_cast<const __m128 *>(ptr + offsets[5]), _MM_MK_INSERTPS_NDX(0,1,0));
    tmp    = _mm_insert_ps(tmp, *reinterpret_cast<const __m128 *>(ptr + offsets[6]), _MM_MK_INSERTPS_NDX(0,2,0));
    tmp    = _mm_insert_ps(tmp, *reinterpret_cast<const __m128 *>(ptr + offsets[7]), _MM_MK_INSERTPS_NDX(0,3,0));
    result = _mm256_insertf128_ps(result, tmp, 1);
#endif
    PERF_END;

    return result;
}
ENABLE_WARNING_UNINITIALIZED

__m256 __attribute__((noinline))
gather_avx_float_set(const float *ptr, const unsigned *offsets)
{
    __m256 result;

    PERF_START;
    result = _mm256_set_ps(ptr[offsets[7]], ptr[offsets[6]],
                           ptr[offsets[5]], ptr[offsets[4]],
                           ptr[offsets[3]], ptr[offsets[2]],
                           ptr[offsets[1]], ptr[offsets[0]]);
    PERF_END;

    return result;
}

__m256 __attribute__((noinline))
gather_avx_float_load(const float *ptr, const unsigned *offsets)
{
    PERF_START;
    float tmp[8] __attribute__((aligned (32))) = { ptr[offsets[0]], ptr[offsets[1]],
                                                   ptr[offsets[2]], ptr[offsets[3]],
                                                   ptr[offsets[4]], ptr[offsets[5]],
                                                   ptr[offsets[6]], ptr[offsets[7]] };
    __m256 result = _mm256_load_ps(tmp);
    PERF_END;

    return result;
}

void __attribute__((noinline))
scatter_avx_float_extract(const __m256& value, float *ptr, const unsigned *offsets)
{
    __m128 tmp;

    PERF_START;
    tmp = _mm256_extractf128_ps(value, 0);
    _MM_EXTRACT_FLOAT(ptr[offsets[0]], tmp, 0);
    _MM_EXTRACT_FLOAT(ptr[offsets[1]], tmp, 1);
    _MM_EXTRACT_FLOAT(ptr[offsets[2]], tmp, 2);
    _MM_EXTRACT_FLOAT(ptr[offsets[3]], tmp, 3);
    tmp = _mm256_extractf128_ps(value, 1);
    _MM_EXTRACT_FLOAT(ptr[offsets[4]], tmp, 0);
    _MM_EXTRACT_FLOAT(ptr[offsets[5]], tmp, 1);
    _MM_EXTRACT_FLOAT(ptr[offsets[6]], tmp, 2);
    _MM_EXTRACT_FLOAT(ptr[offsets[7]], tmp, 3);
    PERF_END;
}

void __attribute__((noinline))
scatter_avx_float_cast(const __m256& value, float *ptr, const unsigned *offsets)
{
    PERF_START;
    const float *p = reinterpret_cast<const float *>(&value);
    ptr[offsets[0]] = p[0];
    ptr[offsets[1]] = p[1];
    ptr[offsets[2]] = p[2];
    ptr[offsets[3]] = p[3];
    ptr[offsets[4]] = p[4];
    ptr[offsets[5]] = p[5];
    ptr[offsets[6]] = p[6];
    ptr[offsets[7]] = p[7];
    PERF_END;
}

void __attribute__((noinline))
scatter_avx_float_store(const __m256& value, float *ptr, const unsigned *offsets)
{
    PERF_START;
    float tmp[8] __attribute__((aligned (32)));
    _mm256_store_ps(tmp, value);
    ptr[offsets[0]] = tmp[0];
    ptr[offsets[1]] = tmp[1];
    ptr[offsets[2]] = tmp[2];
    ptr[offsets[3]] = tmp[3];
    ptr[offsets[4]] = tmp[4];
    ptr[offsets[5]] = tmp[5];
    ptr[offsets[6]] = tmp[6];
    ptr[offsets[7]] = tmp[7];
    PERF_END;
}
#endif

// SSE4 section
#ifdef __SSE4_1__
DISABLE_WARNING_UNINITIALIZED
__m128d __attribute__((noinline))
gather_sse4_double_insert(const double *ptr, const unsigned *offsets)
{
    __m128d result;

    PERF_START;
    result = _mm_loadl_pd(result, ptr + offsets[0]);
    result = _mm_loadh_pd(result, ptr + offsets[1]);
    PERF_END;

    return result;
}
ENABLE_WARNING_UNINITIALIZED

__m128d __attribute__((noinline))
gather_sse4_double_set(const double *ptr, const unsigned *offsets)
{
    __m128d result;

    PERF_START;
    result = _mm_set_pd(ptr[offsets[1]], ptr[offsets[0]]);
    PERF_END;

    return result;
}

__m128d __attribute__((noinline))
gather_sse4_double_load(const double *ptr, const unsigned *offsets)
{
    __m128d result;

    PERF_START;
    double tmp[2] __attribute__((aligned (32))) = { ptr[offsets[0]], ptr[offsets[1]] };
    result = _mm_load_pd(tmp);
    PERF_END;

    return result;
}

void __attribute__((noinline))
scatter_sse4_double_extract(const __m128d& value, double *ptr, const unsigned *offsets)
{
    PERF_START;
    _mm_storel_pd(ptr + offsets[0], value);
    _mm_storeh_pd(ptr + offsets[1], value);
    PERF_END;
}

void __attribute__((noinline))
scatter_sse4_double_cast(const __m128d& value, double *ptr, const unsigned *offsets)
{
    PERF_START;
    const double *p = reinterpret_cast<const double *>(&value);
    ptr[offsets[0]] = p[0];
    ptr[offsets[1]] = p[1];
    PERF_END;
}

void __attribute__((noinline))
scatter_sse4_double_store(const __m128d& value, double *ptr, const unsigned *offsets)
{
    PERF_START;
    double tmp[4] __attribute__((aligned (32)));
    _mm_store_pd(tmp, value);
    ptr[offsets[0]] = tmp[0];
    ptr[offsets[1]] = tmp[1];
    PERF_END;
}

__m128 __attribute__((noinline))
gather_sse4_float_insert(const float *ptr, const unsigned *offsets)
{
    __m128 result;

    PERF_START;
    result = _mm_load_ss(ptr + offsets[0]);
    result = _mm_insert_ps(result, _mm_load_ss(ptr + offsets[1]), _MM_MK_INSERTPS_NDX(0,1,0));
    result = _mm_insert_ps(result, _mm_load_ss(ptr + offsets[2]), _MM_MK_INSERTPS_NDX(0,2,0));
    result = _mm_insert_ps(result, _mm_load_ss(ptr + offsets[3]), _MM_MK_INSERTPS_NDX(0,3,0));
    PERF_END;

    return result;
}

/**
 * Insertps instruction which allows to insert an memory location
 * into a xmm register.
 * Instruction: insertps xmm, xmm/m32, imm8
 *
 * @param a xmm register
 * @param base base pointer
 * @param offset offset
 * @param idx index, has to be a constant number like 0x10, no variable
 */
#define SHORTVEC_INSERT_PS(a, base, offset, idx)                        \
    do {                                                                \
        asm volatile ("insertps %1, (%q2, %q3, 4), %0\n"                \
                      : "+x" (a) : "N" (idx), "r" (base), "r" (offset) : "memory"); \
    } while (0)

__m128 __attribute__((noinline))
gather_sse4_float_insert2(const float *ptr, const unsigned *offsets)
{
    __m128 result;

    PERF_START;
    result = _mm_load_ss(ptr + offsets[0]);
    SHORTVEC_INSERT_PS(result, ptr, offsets[1], _MM_MK_INSERTPS_NDX(0,1,0));
    SHORTVEC_INSERT_PS(result, ptr, offsets[2], _MM_MK_INSERTPS_NDX(0,2,0));
    SHORTVEC_INSERT_PS(result, ptr, offsets[3], _MM_MK_INSERTPS_NDX(0,3,0));
    PERF_END;

    return result;
}

__m128 __attribute__((noinline))
gather_sse4_float_insert3(const float *ptr, const unsigned *offsets)
{
    __m128 result;

    PERF_START;
#if !defined(__clang__) || (__clang_major__ >= 3 && __clang_minor__ >= 5)
    result = _mm_load_ss(ptr + offsets[0]);
    result = _mm_insert_ps(result, *reinterpret_cast<const __m128 *>(ptr + offsets[1]), _MM_MK_INSERTPS_NDX(0,1,0));
    result = _mm_insert_ps(result, *reinterpret_cast<const __m128 *>(ptr + offsets[2]), _MM_MK_INSERTPS_NDX(0,2,0));
    result = _mm_insert_ps(result, *reinterpret_cast<const __m128 *>(ptr + offsets[3]), _MM_MK_INSERTPS_NDX(0,3,0));
#endif
    PERF_END;

    return result;
}

__m128 __attribute__((noinline))
gather_sse4_float_set(const float *ptr, const unsigned *offsets)
{
    __m128 result;

    PERF_START;
    result = _mm_set_ps(ptr[offsets[3]], ptr[offsets[2]],
                        ptr[offsets[1]], ptr[offsets[0]]);
    PERF_END;

    return result;
}

__m128 __attribute__((noinline))
gather_sse4_float_load(const float *ptr, const unsigned *offsets)
{
    __m128 result;

    PERF_START;
    float tmp[4] __attribute__((aligned (32))) = { ptr[offsets[0]], ptr[offsets[1]],
                                                   ptr[offsets[2]], ptr[offsets[3]] };
    result = _mm_load_ps(tmp);
    PERF_END;

    return result;
}

void __attribute__((noinline))
scatter_sse4_float_extract(const __m128& value, float *ptr, const unsigned *offsets)
{
    PERF_START;
    _MM_EXTRACT_FLOAT(ptr[offsets[0]], value, 0);
    _MM_EXTRACT_FLOAT(ptr[offsets[1]], value, 1);
    _MM_EXTRACT_FLOAT(ptr[offsets[2]], value, 2);
    _MM_EXTRACT_FLOAT(ptr[offsets[3]], value, 3);
    PERF_END;
}

// little helper
union ExtractResult {
    int i;
    float f;
};
void __attribute__((noinline))
scatter_sse4_float_extract2(const __m128& value, float *ptr, const unsigned *offsets)
{
    ExtractResult r1, r2, r3, r4;

    PERF_START;
    r1.i = _mm_extract_ps(value, 0);
    r2.i = _mm_extract_ps(value, 1);
    r3.i = _mm_extract_ps(value, 2);
    r4.i = _mm_extract_ps(value, 3);

    ptr[offsets[0]] = r1.f;
    ptr[offsets[1]] = r2.f;
    ptr[offsets[2]] = r3.f;
    ptr[offsets[3]] = r4.f;
    PERF_END;
}

void __attribute__((noinline))
scatter_sse4_float_cast(const __m128& value, float *ptr, const unsigned *offsets)
{
    PERF_START;
    const float *p = reinterpret_cast<const float *>(&value);
    ptr[offsets[0]] = p[0];
    ptr[offsets[1]] = p[1];
    ptr[offsets[2]] = p[2];
    ptr[offsets[3]] = p[3];
    PERF_END;
}

void __attribute__((noinline))
scatter_sse4_float_store(const __m128& value, float *ptr, const unsigned *offsets)
{
    PERF_START;
    float tmp[4] __attribute__((aligned (32)));
    _mm_store_ps(tmp, value);
    ptr[offsets[0]] = tmp[0];
    ptr[offsets[1]] = tmp[1];
    ptr[offsets[2]] = tmp[2];
    ptr[offsets[3]] = tmp[3];
    PERF_END;
}

__m128i __attribute__((noinline))
gather_sse4_int_load(const std::int32_t *ptr, const unsigned *offsets)
{
    __m128i result;
    PERF_START;
    int tmp[4] __attribute__((aligned (32))) = { ptr[offsets[0]], ptr[offsets[1]],
                                                 ptr[offsets[2]], ptr[offsets[3]] };
    result = _mm_load_si128(reinterpret_cast<const __m128i *>(tmp));
    PERF_END;

    return result;
}

__m128i __attribute__((noinline))
gather_sse4_int_set(const std::int32_t *ptr, const unsigned *offsets)
{
    __m128i result;
    PERF_START;
    result = _mm_set_epi32(ptr[offsets[3]], ptr[offsets[2]],
                           ptr[offsets[1]], ptr[offsets[0]]);
    PERF_END;

    return result;
}

DISABLE_WARNING_UNINITIALIZED
__m128i __attribute__((noinline))
gather_sse4_int_insert(const std::int32_t *ptr, const unsigned *offsets)
{
    __m128i result;
    PERF_START;
    result = _mm_insert_epi32(result, ptr[offsets[0]], 0);
    result = _mm_insert_epi32(result, ptr[offsets[1]], 1);
    result = _mm_insert_epi32(result, ptr[offsets[2]], 2);
    result = _mm_insert_epi32(result, ptr[offsets[3]], 3);
    PERF_END;

    return result;
}
ENABLE_WARNING_UNINITIALIZED

void __attribute__((noinline))
scatter_sse4_int_store(const __m128i& value, std::int32_t *ptr, const unsigned *offsets)
{
    PERF_START;
    std::int32_t tmp[4] __attribute__((aligned (16)));
    _mm_store_si128(reinterpret_cast<__m128i *>(tmp), value);
    ptr[offsets[0]] = tmp[0];
    ptr[offsets[1]] = tmp[1];
    ptr[offsets[2]] = tmp[2];
    ptr[offsets[3]] = tmp[3];
    PERF_END;
}

void __attribute__((noinline))
scatter_sse4_int_cast(const __m128i& value, std::int32_t *ptr, const unsigned *offsets)
{
    PERF_START;
    const std::int32_t *p = reinterpret_cast<const std::int32_t *>(&value);
    ptr[offsets[0]] = p[0];
    ptr[offsets[1]] = p[1];
    ptr[offsets[2]] = p[2];
    ptr[offsets[3]] = p[3];
    PERF_END;
}

void __attribute__((noinline))
scatter_sse4_int_extract(const __m128i& value, std::int32_t *ptr, const unsigned *offsets)
{
    PERF_START;
    ptr[offsets[0]] = _mm_extract_epi32(value, 0);
    ptr[offsets[1]] = _mm_extract_epi32(value, 1);
    ptr[offsets[2]] = _mm_extract_epi32(value, 2);
    ptr[offsets[3]] = _mm_extract_epi32(value, 3);
    PERF_END;
}
#endif

// SSE section
#ifdef __SSE2__
DISABLE_WARNING_UNINITIALIZED
__m128d __attribute__((noinline))
gather_sse_double_insert(const double *ptr, const unsigned *offsets)
{
    __m128d result;

    PERF_START;
    result = _mm_loadl_pd(result, ptr + offsets[0]);
    result = _mm_loadh_pd(result, ptr + offsets[1]);
    PERF_END;

    return result;
}
ENABLE_WARNING_UNINITIALIZED

__m128d __attribute__((noinline))
gather_sse_double_set(const double *ptr, const unsigned *offsets)
{
    __m128d result;

    PERF_START;
    result = _mm_set_pd(ptr[offsets[1]], ptr[offsets[0]]);
    PERF_END;

    return result;
}

__m128d __attribute__((noinline))
gather_sse_double_load(const double *ptr, const unsigned *offsets)
{
    __m128d result;

    PERF_START;
    double tmp[2] __attribute__((aligned (32))) = { ptr[offsets[0]], ptr[offsets[1]] };
    result = _mm_load_pd(tmp);
    PERF_END;

    return result;
}

void __attribute__((noinline))
scatter_sse_double_extract(const __m128d& value, double *ptr, const unsigned *offsets)
{
    PERF_START;
    _mm_storel_pd(ptr + offsets[0], value);
    _mm_storeh_pd(ptr + offsets[1], value);
    PERF_END;
}

void __attribute__((noinline))
scatter_sse_double_cast(const __m128d& value, double *ptr, const unsigned *offsets)
{
    PERF_START;
    const double *p = reinterpret_cast<const double *>(&value);
    ptr[offsets[0]] = p[0];
    ptr[offsets[1]] = p[1];
    PERF_END;
}

void __attribute__((noinline))
scatter_sse_double_store(const __m128d& value, double *ptr, const unsigned *offsets)
{
    PERF_START;
    double tmp[4] __attribute__((aligned (32)));
    _mm_store_pd(tmp, value);
    ptr[offsets[0]] = tmp[0];
    ptr[offsets[1]] = tmp[1];
    PERF_END;
}

__m128 __attribute__((noinline))
gather_sse_float_shuffle(const float *ptr, const unsigned *offsets)
{
    __m128 result, f1, f2, f3, f4;

    PERF_START;
    f1 = _mm_load_ss(ptr + offsets[0]);
    f2 = _mm_load_ss(ptr + offsets[1]);
    f3 = _mm_load_ss(ptr + offsets[2]);
    f4 = _mm_load_ss(ptr + offsets[3]);

    f1 = _mm_shuffle_ps(f1, f3, _MM_SHUFFLE(1,0,1,0));
    f2 = _mm_shuffle_ps(f2, f4, _MM_SHUFFLE(0,1,0,1));
    result = _mm_shuffle_ps(f1, f2, _MM_SHUFFLE(3,1,2,0));
    result = _mm_shuffle_ps(result, result, _MM_SHUFFLE(3,1,2,0));
    PERF_END;

    return result;
}

__m128 __attribute__((noinline))
gather_sse_float_unpack(const float *ptr, const unsigned *offsets)
{
    __m128 result, f1, f2, f3, f4;

    PERF_START;
    f1 = _mm_load_ss(ptr + offsets[0]);
    f2 = _mm_load_ss(ptr + offsets[2]);
    // f1: 0 0 3 1
    f1 = _mm_unpacklo_ps(f1, f2);
    f3 = _mm_load_ss(ptr + offsets[1]);
    f4 = _mm_load_ss(ptr + offsets[3]);
    // f3: 0 0 4 2
    f3 = _mm_unpacklo_ps(f3, f4);
    // result: 4 3 2 1
    result = _mm_unpacklo_ps(f1, f3);
    PERF_END;

    return result;
}

__m128 __attribute__((noinline))
gather_sse_float_set(const float *ptr, const unsigned *offsets)
{
    __m128 result;

    PERF_START;
    result = _mm_set_ps(ptr[offsets[3]], ptr[offsets[2]],
                        ptr[offsets[1]], ptr[offsets[0]]);
    PERF_END;

    return result;
}

__m128 __attribute__((noinline))
gather_sse_float_load(const float *ptr, const unsigned *offsets)
{
    __m128 result;

    PERF_START;
    float tmp[4] __attribute__((aligned (32))) = { ptr[offsets[0]], ptr[offsets[1]],
                                                   ptr[offsets[2]], ptr[offsets[3]] };
    result = _mm_load_ps(tmp);
    PERF_END;

    return result;
}

__m128i __attribute__((noinline))
gather_sse_int_load(const std::int32_t *ptr, const unsigned *offsets)
{
    __m128i result;
    PERF_START;
    int tmp[4] __attribute__((aligned (32))) = { ptr[offsets[0]], ptr[offsets[1]],
                                                 ptr[offsets[2]], ptr[offsets[3]] };
    result = _mm_load_si128(reinterpret_cast<const __m128i *>(tmp));
    PERF_END;

    return result;
}

__m128i __attribute__((noinline))
gather_sse_int_set(const std::int32_t *ptr, const unsigned *offsets)
{
    __m128i result;
    PERF_START;
    result = _mm_set_epi32(ptr[offsets[3]], ptr[offsets[2]],
                           ptr[offsets[1]], ptr[offsets[0]]);
    PERF_END;

    return result;
}

DISABLE_WARNING_UNINITIALIZED
__m128i __attribute__((noinline))
gather_sse_int_shift(const std::int32_t *ptr, const unsigned *offsets)
{
    __m128i result;
    PERF_START;
    result = _mm_insert_epi16(result,  ptr[offsets[0]] & 0x0000ffff, 0);
    result = _mm_insert_epi16(result, (ptr[offsets[0]] & 0xffff0000) >> 16, 1);
    result = _mm_insert_epi16(result,  ptr[offsets[1]] & 0x0000ffff, 2);
    result = _mm_insert_epi16(result, (ptr[offsets[1]] & 0xffff0000) >> 16, 3);
    result = _mm_insert_epi16(result,  ptr[offsets[2]] & 0x0000ffff, 4);
    result = _mm_insert_epi16(result, (ptr[offsets[2]] & 0xffff0000) >> 16, 5);
    result = _mm_insert_epi16(result,  ptr[offsets[3]] & 0x0000ffff, 6);
    result = _mm_insert_epi16(result, (ptr[offsets[3]] & 0xffff0000) >> 16, 7);
    PERF_END;

    return result;
}
ENABLE_WARNING_UNINITIALIZED

__m128i __attribute__((noinline))
gather_sse_int_unpack(const std::int32_t *ptr, const unsigned *offsets)
{
    PERF_START;
    __m128i i1 = _mm_cvtsi32_si128(ptr[offsets[0]]);
    __m128i i2 = _mm_cvtsi32_si128(ptr[offsets[1]]);
    __m128i i3 = _mm_cvtsi32_si128(ptr[offsets[2]]);
    __m128i i4 = _mm_cvtsi32_si128(ptr[offsets[3]]);
    i1 = _mm_unpacklo_epi32(i1, i3);
    i3 = _mm_unpacklo_epi32(i2, i4);
    i1 = _mm_unpacklo_epi32(i1, i3);
    PERF_END;

    return i1;
}

void __attribute__((noinline))
scatter_sse_float_shuffle(const __m128& value, float *ptr, const unsigned *offsets)
{
    __m128 tmp;

    PERF_START;
    // 4 3 2 1
    tmp = value;
    _mm_store_ss(ptr + offsets[0], value);
    // 4 3 2 1 - 4 3 2 1 -> 1 4 3 2
    tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
    _mm_store_ss(ptr + offsets[1], tmp);
    tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
    _mm_store_ss(ptr + offsets[2], tmp);
    tmp = _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0,3,2,1));
    _mm_store_ss(ptr + offsets[3], tmp);
    PERF_END;
}

void __attribute__((noinline))
scatter_sse_float_cast(const __m128& value, float *ptr, const unsigned *offsets)
{
    PERF_START;
    const float *p = reinterpret_cast<const float *>(&value);
    ptr[offsets[0]] = p[0];
    ptr[offsets[1]] = p[1];
    ptr[offsets[2]] = p[2];
    ptr[offsets[3]] = p[3];
    PERF_END;
}

void __attribute__((noinline))
scatter_sse_float_store(const __m128& value, float *ptr, const unsigned *offsets)
{
    PERF_START;
    float tmp[4] __attribute__((aligned (32)));
    _mm_store_ps(tmp, value);
    ptr[offsets[0]] = tmp[0];
    ptr[offsets[1]] = tmp[1];
    ptr[offsets[2]] = tmp[2];
    ptr[offsets[3]] = tmp[3];
    PERF_END;
}

void __attribute__((noinline))
scatter_sse_int_store(const __m128i& value, std::int32_t *ptr, const unsigned *offsets)
{
    PERF_START;
    std::int32_t tmp[4] __attribute__((aligned (16)));
    _mm_store_si128(reinterpret_cast<__m128i *>(tmp), value);
    ptr[offsets[0]] = tmp[0];
    ptr[offsets[1]] = tmp[1];
    ptr[offsets[2]] = tmp[2];
    ptr[offsets[3]] = tmp[3];
    PERF_END;
}

void __attribute__((noinline))
scatter_sse_int_cast(const __m128i& value, std::int32_t *ptr, const unsigned *offsets)
{
    PERF_START;
    const std::int32_t *p = reinterpret_cast<const std::int32_t *>(&value);
    ptr[offsets[0]] = p[0];
    ptr[offsets[1]] = p[1];
    ptr[offsets[2]] = p[2];
    ptr[offsets[3]] = p[3];
    PERF_END;
}

void __attribute__((noinline))
scatter_sse_int_convert(const __m128i& value, std::int32_t *ptr, const unsigned *offsets)
{
    PERF_START;
    ptr[offsets[0]] = _mm_cvtsi128_si32(value);
    ptr[offsets[1]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(value, _MM_SHUFFLE(0,3,2,1)));
    ptr[offsets[2]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(value, _MM_SHUFFLE(1,0,3,2)));
    ptr[offsets[3]] = _mm_cvtsi128_si32(_mm_shuffle_epi32(value, _MM_SHUFFLE(2,1,0,3)));
    PERF_END;
}

void __attribute__((noinline))
scatter_sse_int_shift(const __m128i& value, std::int32_t *ptr, const unsigned *offsets)
{
    PERF_START;
    ptr[offsets[0]] = _mm_extract_epi16(value, 0) | (_mm_extract_epi16(value, 1) << 16);
    ptr[offsets[1]] = _mm_extract_epi16(value, 2) | (_mm_extract_epi16(value, 3) << 16);
    ptr[offsets[2]] = _mm_extract_epi16(value, 4) | (_mm_extract_epi16(value, 5) << 16);
    ptr[offsets[3]] = _mm_extract_epi16(value, 6) | (_mm_extract_epi16(value, 7) << 16);
    PERF_END;
}
#endif

int main(int argc, char *argv[])
{
    // dummy main: prevent compiler from optimizing too much away
    const int LEN = 20;
    std::string methods;

    // do stuff, based on compiler settings
#if defined (__AVX512F__)
    methods = "SSE2 SSE4 AVX AVX2 AVX512";
#endif
#if defined(__AVX2__)
    methods = "SSE2 SSE4 AVX AVX2";
#endif
#if defined(__AVX__) && !defined(__AVX2__) && !defined(__AVX512F__)
    methods = "SSE2 SSE4 AVX";
#endif
#if defined(__SSE4_1__) && !defined(__AVX__) && !defined(__AVX2__) && !defined(__AVX512F__)
    methods = "SSE2 SSE4";
#endif
#if defined(__SSE2__) && !defined(__SSE4_1__) && !defined(__AVX__) && !defined(__AVX2__) && !defined(__AVX512F__)
    methods = "SSE2";
#endif

    std::cout << "Gather Test for " << methods << "..." << std::endl;

    // get some user data
    float floats[LEN];
    double doubles[LEN];
    std::int32_t ints[LEN];
    for (unsigned i = 0; i < LEN; ++i) {
        double tmp;
        if (!(std::cin >> tmp)) {
            std::cerr << "Failed to get number!" << std::endl;
            return EXIT_FAILURE;
        }
        doubles[i] = tmp;
        floats[i] = static_cast<float>(tmp);
        ints[i] = static_cast<std::int32_t>(tmp);
    }
    // indices!
    unsigned idx[] __attribute__ ((aligned (64)))
        = { 5, 4, 8, 11, 18, 17, 0, 12, 3, 19, 1, 7, 2, 9, 10, 16 };
    // scatter
    float fscatter[LEN];
    double dscatter[LEN];
    std::int32_t iscatter[LEN];

    // call some methods: first of all, call sse methods...
#ifdef __SSE2__
    {
        __m128d r1  = gather_sse_double_insert(doubles, idx);
        __m128d r2  = gather_sse_double_load(doubles, idx);
        __m128d r3  = gather_sse_double_set(doubles, idx);
        __m128  r4  = gather_sse_float_shuffle(floats, idx);
        __m128  r5  = gather_sse_float_unpack(floats, idx);
        __m128  r6  = gather_sse_float_set(floats, idx);
        __m128  r7  = gather_sse_float_load(floats, idx);
        __m128i r8  = gather_sse_int_load(ints, idx);
        __m128i r9  = gather_sse_int_set(ints, idx);
        __m128i r10 = gather_sse_int_shift(ints, idx);
        __m128i r11 = gather_sse_int_unpack(ints, idx);
        std::cout << "SSE2: Gather: Double" << std::endl;
        print_vec_sse(r1);
        print_vec_sse(r2);
        print_vec_sse(r3);
        std::cout << "SSE2: Gather: Float" << std::endl;
        print_vec_sse(r4);
        print_vec_sse(r5);
        print_vec_sse(r6);
        print_vec_sse(r7);
        std::cout << "SSE2: Gather: Int32" << std::endl;
        print_vec_sse(r8);
        print_vec_sse(r9);
        print_vec_sse(r10);
        print_vec_sse(r11);
        std::cout << "SSE2: Scatter: Double" << std::endl;
        SCATTER_DOUBLE(r1, scatter_sse_double_extract);
        SCATTER_DOUBLE(r1, scatter_sse_double_cast);
        SCATTER_DOUBLE(r1, scatter_sse_double_store);
        std::cout << "SSE2: Scatter: Float" << std::endl;
        SCATTER_FLOAT(r4, scatter_sse_float_shuffle);
        SCATTER_FLOAT(r4, scatter_sse_float_cast);
        SCATTER_FLOAT(r4, scatter_sse_float_store);
        std::cout << "SSE2: Scatter: Int32" << std::endl;
        SCATTER_INT32(r8, scatter_sse_int_store);
        SCATTER_INT32(r8, scatter_sse_int_cast);
        SCATTER_INT32(r8, scatter_sse_int_convert);
        SCATTER_INT32(r8, scatter_sse_int_shift);
    }
#endif

#ifdef __SSE4_1__
    {
        __m128d r1  = gather_sse4_double_insert(doubles, idx);
        __m128d r2  = gather_sse4_double_load(doubles, idx);
        __m128d r3  = gather_sse4_double_set(doubles, idx);
        __m128  r4  = gather_sse4_float_insert(floats, idx);
        __m128  r5  = gather_sse4_float_insert2(floats, idx);
        __m128  r6  = gather_sse4_float_insert3(floats, idx);
        __m128  r7  = gather_sse4_float_set(floats, idx);
        __m128  r8  = gather_sse4_float_load(floats, idx);
        __m128i r9  = gather_sse4_int_load(ints, idx);
        __m128i r10 = gather_sse4_int_set(ints, idx);
        __m128i r11 = gather_sse4_int_insert(ints, idx);
        std::cout << "SSE4: Gather: Double" << std::endl;
        print_vec_sse(r1);
        print_vec_sse(r2);
        print_vec_sse(r3);
        std::cout << "SSE4: Gather: Float" << std::endl;
        print_vec_sse(r4);
        print_vec_sse(r5);
        print_vec_sse(r6);
        print_vec_sse(r7);
        print_vec_sse(r8);
        std::cout << "SSE4: Gather: Int32" << std::endl;
        print_vec_sse(r9);
        print_vec_sse(r10);
        print_vec_sse(r11);
        std::cout << "SSE4: Scatter: Double" << std::endl;
        SCATTER_DOUBLE(r1, scatter_sse4_double_extract);
        SCATTER_DOUBLE(r1, scatter_sse4_double_cast);
        SCATTER_DOUBLE(r1, scatter_sse4_double_store);
        std::cout << "SSE4: Scatter: Float" << std::endl;
        SCATTER_FLOAT(r4, scatter_sse4_float_extract);
        SCATTER_FLOAT(r4, scatter_sse4_float_extract2);
        SCATTER_FLOAT(r4, scatter_sse4_float_cast);
        SCATTER_FLOAT(r4, scatter_sse4_float_store);
        std::cout << "SSE4: Scatter: Int32" << std::endl;
        SCATTER_INT32(r9, scatter_sse4_int_cast);
        SCATTER_INT32(r9, scatter_sse4_int_store);
        SCATTER_INT32(r9, scatter_sse4_int_extract);
    }
#endif

#ifdef __AVX__
    {
        __m256d r1 = gather_avx_double_insert(doubles, idx);
        __m256d r2 = gather_avx_double_load(doubles, idx);
        __m256d r3 = gather_avx_double_set(doubles, idx);
        __m256  r4 = gather_avx_float_insert(floats, idx);
        __m256  r5 = gather_avx_float_insert2(floats, idx);
        __m256  r6 = gather_avx_float_insert3(floats, idx);
        __m256  r7 = gather_avx_float_insert4(floats, idx);
        __m256  r8 = gather_avx_float_set(floats, idx);
        __m256  r9 = gather_avx_float_load(floats, idx);
        std::cout << "AVX: Gather: Double" << std::endl;
        print_vec_avx(r1);
        print_vec_avx(r2);
        print_vec_avx(r3);
        std::cout << "AVX: Gather: Float" << std::endl;
        print_vec_avx(r4);
        print_vec_avx(r5);
        print_vec_avx(r6);
        print_vec_avx(r7);
        print_vec_avx(r8);
        print_vec_avx(r9);
        std::cout << "AVX: Scatter: Double" << std::endl;
        SCATTER_DOUBLE(r1, scatter_avx_double_extract);
        SCATTER_DOUBLE(r1, scatter_avx_double_cast);
        SCATTER_DOUBLE(r1, scatter_avx_double_store);
        std::cout << "AVX: Scatter: Float" << std::endl;
        SCATTER_FLOAT(r4, scatter_avx_float_extract);
        SCATTER_FLOAT(r4, scatter_avx_float_cast);
        SCATTER_FLOAT(r4, scatter_avx_float_store);
    }
#endif

#ifdef __AVX2__
    {
        __m256d r1 = gather_avx2_double_gather(doubles, idx);
        __m256  r2 = gather_avx2_float_gather(floats, idx);
        std::cout << "AVX2: Gather: Double" << std::endl;
        print_vec_avx(r1);
        std::cout << "AVX2: Gather: Float" << std::endl;
        print_vec_avx(r2);
    }
#endif

#ifdef __AVX512F__
    {
        __m512d r1 = gather_avx512_double_gather(doubles, idx);
        __m512  r2 = gather_avx512_float_gather(floats, idx);
        std::cout << "AVX512: Gather: Double" << std::endl;
        print_vec_avx512(r1);
        std::cout << "AVX512: Gather: Float" << std::endl;
        print_vec_avx512(r2);
        std::cout << "AVX512: Scatter: Double" << std::endl;
        SCATTER_DOUBLE(r1, scatter_avx512_double_scatter);
        std::cout << "AVX512: Scatter: Float" << std::endl;
        SCATTER_FLOAT(r2, scatter_avx512_float_scatter);
    }
#endif

    return EXIT_SUCCESS;
}
