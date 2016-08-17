/*
 * sha512 djm34
 *
 */

/*
 * sha-512 kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014  djm34
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author   phm <phm@inbox.com>
 */
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "cuda_helper.h"
#define SPH_C64(x)    ((uint64_t)(x ## ULL))

// aus heavy.cu
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);
extern int device_major[8];

__constant__ uint64_t c_PaddedMessage80[16];

static __constant__ uint64_t H_512[8] = {
	SPH_C64(0x6A09E667F3BCC908), SPH_C64(0xBB67AE8584CAA73B),
	SPH_C64(0x3C6EF372FE94F82B), SPH_C64(0xA54FF53A5F1D36F1),
	SPH_C64(0x510E527FADE682D1), SPH_C64(0x9B05688C2B3E6C1F),
	SPH_C64(0x1F83D9ABFB41BD6B), SPH_C64(0x5BE0CD19137E2179)
};

static __constant__ uint64_t const gpu_WK[80] = {
	0x428a2f98d728ae22,0x7137449123ef65cd,0xb5c0fbcfec4d3b2f,0xe9b5dba58189dbbc,
	0x3956c25bf348b538,0x59f111f1b605d019,0x923f82a4af194f9b,0xab1c5ed5da6d8118,
	0xd807aa98a3030242,0x12835b0145706fbe,0x243185be4ee4b28c,0x550c7dc3d5ffb4e2,
	0x72be5d74f27b896f,0x80deb1fe3b1696b1,0x9bdc06a725c71235,0xc19bf174cf692a64,
	0xe49b69c19ef14ad2,0xf0384786384f4472,0xfc19dc68b8cd5b5,0x240ca9dbb7ad9067,
	0x2de92c6f592b0275,0x68f48504aaad3043,0x5cb0a9dcbd41ffa4,0xb3fb89db0bda5464,
	0x99325152ee671cc9,0x7123ae0d31ff6272,0xb0033ff658fdfd45,0xe1fa23485dabbe42,
	0x40e00d5c2dc22ec2,0x560bec2676fa4690,0xf5d468548bee0586,0x7eb60f7f9b748ba,
	0x2b757e0560795790,0xb52d238794343086,0x6adcfb55ae70f872,0x6398611e8a733079,
	0x291459ab496a27f0,0x80c473236c425024,0x232854f0097e8487,0xf30368b14ad940b2,
	0x4505c8f72220c3ac,0x79223106d06018ab,0xa92a9358ef68a70,0x2213ed04d9d16738,
	0xfde1eec528488436,0x1b8e578b3cb41cb3,0xbb29a10ec7f0e115,0x251bea9790505f29,
	0xd4457f79ff355b71,0x2e05d9dbe2066b35,0xb85b71b919d5f399,0xcc59173fcaca449c,
	0x2722a9a858047e9d,0xf102d1898b991e4e,0xf0a86da2960a222b,0x7dce8dba654ef680,
	0x5872e4ced838147c,0x81a93d5212186f4e,0x9b18b478d47f8e8c,0x4d8f8e0e5d851bf6,
	0xcfabc63ef8d4741b,0xa11c143919b22c32,0x743e8af79927636c,0x3d0d0f5b5bc98d7a,
	0x297c0e410c9e6c3c,0x932ee100a01733b3,0xf5b1fd3afc9cd585,0xd8b62f8c1408fe3a,
	0x7192616861de6cf8,0xb4e118d6ac3895b6,0xffd847e31993915c,0x3022c96b9e13ee56,
	0xe00cac4914baa991,0xa8f14913ade567aa,0x2021caba7e45a5a6,0x534bdc6351b491af,
	0x4fdbfb25d8e33212,0xe320be3f9eababf3,0xa3ef683366aff9d0,0xb3ef91a4e5db6e75
};

static __constant__
#if __CUDA_ARCH__ > 500
__align__(16)
#else
__align__(8)
#endif
uint64_t K_512[80] = {
	0x428A2F98D728AE22, 0x7137449123EF65CD, 0xB5C0FBCFEC4D3B2F, 0xE9B5DBA58189DBBC,
	0x3956C25BF348B538, 0x59F111F1B605D019, 0x923F82A4AF194F9B, 0xAB1C5ED5DA6D8118,
	0xD807AA98A3030242, 0x12835B0145706FBE, 0x243185BE4EE4B28C, 0x550C7DC3D5FFB4E2,
	0x72BE5D74F27B896F, 0x80DEB1FE3B1696B1, 0x9BDC06A725C71235, 0xC19BF174CF692694,
	0xE49B69C19EF14AD2, 0xEFBE4786384F25E3, 0x0FC19DC68B8CD5B5, 0x240CA1CC77AC9C65,
	0x2DE92C6F592B0275, 0x4A7484AA6EA6E483, 0x5CB0A9DCBD41FBD4, 0x76F988DA831153B5,
	0x983E5152EE66DFAB, 0xA831C66D2DB43210, 0xB00327C898FB213F, 0xBF597FC7BEEF0EE4,
	0xC6E00BF33DA88FC2, 0xD5A79147930AA725, 0x06CA6351E003826F, 0x142929670A0E6E70,
	0x27B70A8546D22FFC, 0x2E1B21385C26C926, 0x4D2C6DFC5AC42AED, 0x53380D139D95B3DF,
	0x650A73548BAF63DE, 0x766A0ABB3C77B2A8, 0x81C2C92E47EDAEE6, 0x92722C851482353B,
	0xA2BFE8A14CF10364, 0xA81A664BBC423001, 0xC24B8B70D0F89791, 0xC76C51A30654BE30,
	0xD192E819D6EF5218, 0xD69906245565A910, 0xF40E35855771202A, 0x106AA07032BBD1B8,
	0x19A4C116B8D2D0C8, 0x1E376C085141AB53, 0x2748774CDF8EEB99, 0x34B0BCB5E19B48A8,
	0x391C0CB3C5C95A63, 0x4ED8AA4AE3418ACB, 0x5B9CCA4F7763E373, 0x682E6FF3D6B2B8A3,
	0x748F82EE5DEFB2FC, 0x78A5636F43172F60, 0x84C87814A1F0AB72, 0x8CC702081A6439EC,
	0x90BEFFFA23631E28, 0xA4506CEBDE82BDE9, 0xBEF9A3F7B2C67915, 0xC67178F2E372532B,
	0xCA273ECEEA26619C, 0xD186B8C721C0C207, 0xEADA7DD6CDE0EB1E, 0xF57D4F7FEE6ED178,
	0x06F067AA72176FBA, 0x0A637DC5A2C898A6, 0x113F9804BEF90DAE, 0x1B710B35131C471B,
	0x28DB77F523047D84, 0x32CAAB7B40C72493, 0x3C9EBE0A15C9BEBC, 0x431D67C49C100D4C,
	0x4CC5D4BECB3E42B6, 0x597F299CFC657E2A, 0x5FCB6FAB3AD6FAEC, 0x6C44198C4A475817
};


static __device__ __forceinline__ uint64_t bsg5_0(uint64_t x)
{
	uint64_t r1 = ROTR64(x,28);
	uint64_t r2 = ROTR64(x,34);
	uint64_t r3 = ROTR64(x,39);
	return xor3(r1,r2,r3);
}

static __device__ __forceinline__ uint64_t bsg5_1(uint64_t x)
{
	uint64_t r1 = ROTR64(x,14);
	uint64_t r2 = ROTR64(x,18);
	uint64_t r3 = ROTR64(x,41);
	return xor3(r1,r2,r3);
}

static __device__ __forceinline__ uint64_t ssg5_0(uint64_t x)
{
	uint64_t r1 = ROTR64(x,1);
	uint64_t r2 = ROTR64(x,8);
	uint64_t r3 = shr_t64(x,7);
	return xor3(r1,r2,r3);
}

static __device__ __forceinline__ uint64_t ssg5_1(uint64_t x)
{
	uint64_t r1 = ROTR64(x,19);
	uint64_t r2 = ROTR64(x,61);
	uint64_t r3 = shr_t64(x,6);
	return xor3(r1,r2,r3);
}


static __device__ __forceinline__ void sha3_step2(uint64_t* r,uint64_t* W,uint64_t* K,int ord,int i)
{
int u = 8-ord;
uint64_t a=r[(0+u)& 7];
uint64_t b=r[(1+u)& 7];
uint64_t c=r[(2+u)& 7];
uint64_t d=r[(3+u)& 7];
uint64_t e=r[(4+u)& 7];
uint64_t f=r[(5+u)& 7];
uint64_t g=r[(6+u)& 7];
uint64_t h=r[(7+u)& 7];

uint64_t T1, T2;
T1 = h+bsg5_1(e)+xandx64(e,f,g)+W[i]+K[i];
T2 = bsg5_0(a) + andor(a,b,c);
r[(3+u)& 7] = d + T1;
r[(7+u)& 7] = T1 + T2;

}


static __device__ __forceinline__ void sha3_step3(uint64_t* r,const uint64_t* W,int ord,int i)
{
	int u = 8-ord;
	uint64_t a=r[(0+u)& 7];
	uint64_t b=r[(1+u)& 7];
	uint64_t c=r[(2+u)& 7];
	uint64_t d=r[(3+u)& 7];
	uint64_t e=r[(4+u)& 7];
	uint64_t f=r[(5+u)& 7];
	uint64_t g=r[(6+u)& 7];
	uint64_t h=r[(7+u)& 7];

	uint64_t T1, T2;
	T1 = h+bsg5_1(e)+xandx64(e,f,g)+W[i];
	T2 = bsg5_0(a) + andor(a,b,c);
	r[(3+u)& 7] = d + T1;
	r[(7+u)& 7] = T1 + T2;
}


__global__ void sha512_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint64_t *inpHash = (uint64_t*)&g_hash + 8*thread;

		uint64_t W[80];
        uint64_t r[8];
#pragma unroll 71
		for (int i=9;i<80;i++) {W[i]=0;}

#pragma unroll 8
 		for (int i = 0; i < 8; i ++) {
			W[i] = cuda_swab64(inpHash[i]);
			r[i] = H_512[i];}

		W[8] = 0x8000000000000000;
		W[15]= 0x0000000000000200;
#pragma unroll 64
		for (int i = 16; i < 80; i ++)
 			W[i] = sph_t64(ssg5_1(W[i - 2]) + W[i - 7] + ssg5_0(W[i - 15]) + W[i - 16]);

#if __CUDA_ARCH__ < 500    // go figure...
#pragma unroll 10
#endif
		for (int i = 0; i < 10; i ++) {
#pragma unroll 8
			for (int ord=0;ord<8;ord++) {sha3_step2(r,W,K_512,ord,8*i+ord);}
		}

#pragma unroll 8
		for (int i = 0; i < 8; i++) {r[i] = sph_t64(r[i] + H_512[i]);}

      #pragma unroll 8
      for (int u = 0; u < 8; u ++)
            inpHash[u] = cuda_swab64(r[u]);
	}
}


__global__ void __launch_bounds__(256,3) m7_sha512_gpu50_hash_120(int threads, uint32_t startNounce, uint64_t *outputHash)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
			uint32_t nounce = startNounce + thread;

		uint64_t W[80];
        uint64_t r[8];
#pragma unroll 8
		for (int i = 0; i < 8; i ++) {r[i] = H_512[i];}
#pragma unroll 14
		for (int i = 0; i < 14; i ++) {W[i] = cuda_swab64(c_PaddedMessage80[i]);}
		    W[14] =  cuda_swab64(REPLACE_HIWORD(c_PaddedMessage80[14],nounce));
            W[15] =  cuda_swab64(c_PaddedMessage80[15]);

#pragma unroll 64
		for (int i = 16; i < 80; i ++)
 			W[i] = sph_t64(ssg5_1(W[i - 2]) + W[i - 7] + ssg5_0(W[i - 15]) + W[i - 16]);

#if __CUDA_ARCH__ < 500    // go figure...
#pragma unroll 10
#endif
		for (int i = 0; i < 10; i ++) {
#pragma unroll 8
			for (int ord=0;ord<8;ord++) {sha3_step2(r,W,K_512,ord,8*i+ord); }
		}
 uint64_t tempr[8];
#pragma unroll 8
		for (int i = 0; i < 8; i++) {tempr[i] = r[i] = sph_t64(r[i] + H_512[i]);}


#if __CUDA_ARCH__ < 500    // go figure...
#pragma unroll
#endif
		for (int i = 0; i < 10; i ++) {
#pragma unroll 8
			for (int ord=0;ord<8;ord++) {sha3_step3(r,gpu_WK,ord,8*i+ord); }
		}

#pragma unroll 8
for(int i=0;i<8;i++) {outputHash[i*threads+thread] = cuda_swab64(sph_t64(r[i] + tempr[i]));}

 } /// thread
}


__global__ void __launch_bounds__(256,4) m7_sha512_gpu_hash_120(int threads, uint32_t startNounce, uint64_t *outputHash)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
			uint32_t nounce = startNounce + thread;

		uint64_t W[80];
        uint64_t r[8];
#pragma unroll 8
		for (int i = 0; i < 8; i ++) {r[i] = H_512[i];}
#pragma unroll 14
		for (int i = 0; i < 14; i ++) {W[i] = cuda_swab64(c_PaddedMessage80[i]);}
		    W[14] =  cuda_swab64(REPLACE_HIWORD(c_PaddedMessage80[14],nounce));
            W[15] =  cuda_swab64(c_PaddedMessage80[15]);

#pragma unroll 64
		for (int i = 16; i < 80; i ++)
 			W[i] = sph_t64(ssg5_1(W[i - 2]) + W[i - 7] + ssg5_0(W[i - 15]) + W[i - 16]);

#if __CUDA_ARCH__ < 500    // go figure...
#pragma unroll 10
#endif
		for (int i = 0; i < 10; i ++) {
#pragma unroll 8
			for (int ord=0;ord<8;ord++) {sha3_step2(r,W,K_512,ord,8*i+ord); }
		}
 uint64_t tempr[8];
#pragma unroll 8
		for (int i = 0; i < 8; i++) {tempr[i] = r[i] = sph_t64(r[i] + H_512[i]);}


#if __CUDA_ARCH__ < 500    // go figure...
#pragma unroll
#endif
		for (int i = 0; i < 10; i ++) {
#pragma unroll 8
			for (int ord=0;ord<8;ord++) {sha3_step3(r,gpu_WK,ord,8*i+ord); }
		}

#pragma unroll 8
for(int i=0;i<8;i++) {outputHash[i*threads+thread] = cuda_swab64(sph_t64(r[i] + tempr[i]));}

	} /// thread
}


void sha512_cpu_init(int thr_id, int threads)
{
}


__host__ void sha512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{

	const int threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);
	sha512_gpu_hash_64<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);

	MyStreamSynchronize(NULL, order, thr_id);
}


__host__ void sha512_setBlock_120(void *pdata)
{
	unsigned char PaddedMessage[128];
	uint8_t ending =0x80;
	memcpy(PaddedMessage, pdata, 122);
	memset(PaddedMessage+122,ending,1);
	memset(PaddedMessage+123, 0, 5); //useless
	cudaMemcpyToSymbol( c_PaddedMessage80, PaddedMessage, 16*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);

}

__host__ void m7_sha512_cpu_hash_120(int thr_id, int threads, uint32_t startNounce, uint64_t *d_outputHash, int order)
{

	const int threadsperblock = 256;

	dim3 grid(threads/threadsperblock);
	dim3 block(threadsperblock);
	if (device_major[thr_id]==5) m7_sha512_gpu50_hash_120<<<grid, block>>>(threads, startNounce, d_outputHash);
	else m7_sha512_gpu_hash_120<<<grid, block>>>(threads, startNounce, d_outputHash);

	MyStreamSynchronize(NULL, order, thr_id);
}

