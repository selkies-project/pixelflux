/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// ARGB/ABGR -> NV12 with BT.601 limited-range coefficients, chroma averaged over each 2x2 block.
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
        unsigned char* luma = dst + (long)dst_pitch * y;
        for (int dx = 0; dx < 2; ++dx) {
            int x = min(cx * 2 + dx, width - 1);
            const unsigned char* px = row + (long)x * 4;
            float b = px[swap_rb ? 2 : 0], g = px[1], r = px[swap_rb ? 0 : 2];
            sr += r; sg += g; sb += b;
            float yf = 0.299f * r + 0.587f * g + 0.114f * b;
            luma[x] = (unsigned char)__float2int_rn(16.0f + yf * (219.0f / 255.0f));
        }
    }
    sr *= 0.25f; sg *= 0.25f; sb *= 0.25f;
    float yf = 0.299f * sr + 0.587f * sg + 0.114f * sb;
    float cb = 128.0f + (sb - yf) * (224.0f / 255.0f) / (2.0f * (1.0f - 0.114f));
    float cr = 128.0f + (sr - yf) * (224.0f / 255.0f) / (2.0f * (1.0f - 0.299f));
    uv[0] = (unsigned char)__float2int_rn(fminf(fmaxf(cb, 0.0f), 255.0f));
    uv[1] = (unsigned char)__float2int_rn(fminf(fmaxf(cr, 0.0f), 255.0f));
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
        unsigned char* luma = dst + (long)dst_pitch * y;
        for (int dx = 0; dx < 2; ++dx) {
            int x = min(cx * 2 + dx, width - 1);
            uchar4 px = tex2D<uchar4>(src, x, y);
            float b = swap_rb ? px.z : px.x, g = px.y, r = swap_rb ? px.x : px.z;
            sr += r; sg += g; sb += b;
            float yf = 0.299f * r + 0.587f * g + 0.114f * b;
            luma[x] = (unsigned char)__float2int_rn(16.0f + yf * (219.0f / 255.0f));
        }
    }
    sr *= 0.25f; sg *= 0.25f; sb *= 0.25f;
    float yf = 0.299f * sr + 0.587f * sg + 0.114f * sb;
    float cb = 128.0f + (sb - yf) * (224.0f / 255.0f) / (2.0f * (1.0f - 0.114f));
    float cr = 128.0f + (sr - yf) * (224.0f / 255.0f) / (2.0f * (1.0f - 0.299f));
    uv[0] = (unsigned char)__float2int_rn(fminf(fmaxf(cb, 0.0f), 255.0f));
    uv[1] = (unsigned char)__float2int_rn(fminf(fmaxf(cr, 0.0f), 255.0f));
}
