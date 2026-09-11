/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// ARGB/ABGR -> NV12, BT.709 at limited range. NVENC's own conversion follows the matrix a
// session declares but weights the two columns of a 4:2:0 block 3:1 instead of averaging them,
// which is what these kernels replace.

__device__ __forceinline__ float luma(float r, float g, float b)
{
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

__device__ __forceinline__ unsigned char clamp8(float v)
{
    return (unsigned char)__float2int_rn(fminf(fmaxf(v, 0.0f), 255.0f));
}

__device__ __forceinline__ unsigned char luma8(float r, float g, float b)
{
    return clamp8(16.0f + luma(r, g, b) * (219.0f / 255.0f));
}

// The chroma pair of a 2x2 block's average RGB.
__device__ __forceinline__ void chroma8(float r, float g, float b, unsigned char* cb, unsigned char* cr)
{
    float y = luma(r, g, b);
    *cb = clamp8(128.0f + (b - y) * (224.0f / 255.0f) / (2.0f * (1.0f - 0.0722f)));
    *cr = clamp8(128.0f + (r - y) * (224.0f / 255.0f) / (2.0f * (1.0f - 0.2126f)));
}

// A pixel's B, G and R, from either byte order.
__device__ __forceinline__ void unpack(const unsigned char* px, int swap_rb, float* r, float* g, float* b)
{
    *b = px[swap_rb ? 2 : 0];
    *g = px[1];
    *r = px[swap_rb ? 0 : 2];
}

extern "C" __global__ void argb_to_nv12(
    const unsigned char* __restrict__ src, int src_pitch,
    unsigned char* __restrict__ dst, int dst_pitch,
    int width, int height, int swap_rb)
{
    int cx = blockIdx.x * blockDim.x + threadIdx.x;   // chroma column
    int cy = blockIdx.y * blockDim.y + threadIdx.y;   // chroma row
    if (cx * 2 >= width || cy * 2 >= height) return;

    float sr = 0.0f, sg = 0.0f, sb = 0.0f;
    unsigned char* uv = dst + (long)dst_pitch * height + (long)dst_pitch * cy + cx * 2;
    for (int dy = 0; dy < 2; ++dy) {
        int y = min(cy * 2 + dy, height - 1);
        const unsigned char* row = src + (long)src_pitch * y;
        unsigned char* luma_row = dst + (long)dst_pitch * y;
        for (int dx = 0; dx < 2; ++dx) {
            int x = min(cx * 2 + dx, width - 1);
            float r, g, b;
            unpack(row + (long)x * 4, swap_rb, &r, &g, &b);
            sr += r; sg += g; sb += b;
            luma_row[x] = luma8(r, g, b);
        }
    }
    chroma8(sr * 0.25f, sg * 0.25f, sb * 0.25f, &uv[0], &uv[1]);
}

// The same convert for an import the driver hands back as a CUDA array rather than linear
// memory: the texture unit resolves the array's layout, so the RGB is read in place instead of
// being copied into a linear surface first.
extern "C" __global__ void argb_tex_to_nv12(
    cudaTextureObject_t src,
    unsigned char* __restrict__ dst, int dst_pitch,
    int width, int height, int swap_rb)
{
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx * 2 >= width || cy * 2 >= height) return;

    float sr = 0.0f, sg = 0.0f, sb = 0.0f;
    unsigned char* uv = dst + (long)dst_pitch * height + (long)dst_pitch * cy + cx * 2;
    for (int dy = 0; dy < 2; ++dy) {
        int y = min(cy * 2 + dy, height - 1);
        unsigned char* luma_row = dst + (long)dst_pitch * y;
        for (int dx = 0; dx < 2; ++dx) {
            int x = min(cx * 2 + dx, width - 1);
            uchar4 px = tex2D<uchar4>(src, x, y);
            float b = swap_rb ? px.z : px.x, g = px.y, r = swap_rb ? px.x : px.z;
            sr += r; sg += g; sb += b;
            luma_row[x] = luma8(r, g, b);
        }
    }
    chroma8(sr * 0.25f, sg * 0.25f, sb * 0.25f, &uv[0], &uv[1]);
}
