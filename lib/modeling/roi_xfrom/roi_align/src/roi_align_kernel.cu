#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "roi_align_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

    /*** Forward ***/

    __device__ float bilinear_interpolate(const float* bottom_data, const int height, const int width,
                                          float y, float x, const int index /* index for debug only*/) {
            // deal with cases that inverse elements are out of feature map boundary
            if (y < -1.0 || y > height || x < -1.0 || x > width) {
                // empty
                return 0;
            }

            if (y <= 0) {
                y = 0;
            }
            if (x <= 0) {
                x = 0;
            }
            
            int y_low = (int)y;
            int x_low = (int)x;
            int y_high;
            int x_high;
            
            if (y_low >= height - 1) {
                y_high = y_low = height - 1;
                y = (float)y_low;
            } else {
                y_high = y_low + 1;
            }
            
            if (x_low >= width - 1) {
                x_high = x_low = width - 1;
                x = (float)x_low;
            } else {
                x_high = x_low + 1;
            }
            
            float ly = y - y_low;
            float lx = x - x_low;
            float hy = 1. -ly, hx = 1. - lx;
            // do bilinear interpolation
            float v1 = bottom_data[y_low * width + x_low];
            float v2 = bottom_data[y_low * width + x_high];
            float v3 = bottom_data[y_high * width + x_low];
            float v4 = bottom_data[y_high * width + x_high];
            float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

            float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

            return val;
        }

    __global__ void ROIAlignForward(const int nthreads, const float* bottom_data, const float spatial_scale, const int height, const int width,
                                    const int channels, const int aligned_height, const int aligned_width, const int sampling_ratio,
                                    const float* bottom_rois, float* top_data) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            // (n, c, ph, pw) is an element in the aligned output
            int pw = index % aligned_width;
            int ph = (index / aligned_width) % aligned_height;
            int c  = (index / aligned_width / aligned_height) % channels;
            int n  = index / aligned_width / aligned_height / channels;

            const float* offset_bottom_rois = bottom_rois + n * 5;
            int roi_batch_ind = offset_bottom_rois[0];

            // Do not using rounding; this implementation detail is critical
            float roi_start_w = offset_bottom_rois[1] * spatial_scale;
            float roi_start_h = offset_bottom_rois[2] * spatial_scale;
            float roi_end_w = offset_bottom_rois[3] * spatial_scale;
            float roi_end_h = offset_bottom_rois[4] * spatial_scale;

            // Force malformed ROIs to be 1x1
            float roi_width = fmaxf(roi_end_w - roi_start_w, 1.f);
            float roi_height = fmaxf(roi_end_h - roi_start_h, 1.f);
            float bin_size_h = roi_height / aligned_height;
            float bin_size_w = roi_width / aligned_width;

            const float* offset_bottom_data =
                bottom_data + (roi_batch_ind * channels + c) * height * width;

            // We use roi_bin_grid to sample the grid and mimic integral
            int roi_bin_grid_h = (sampling_ratio > 0)
                ? sampling_ratio
                : ceil(roi_height / aligned_height); // e.g., = 2
            int roi_bin_grid_w =
                (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / aligned_width);

            // We do average (integral) pooling inside a bin
            const float count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

            float output_val = 0.;
            for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
            {
                const float y = roi_start_h + ph * bin_size_h +
                    (iy + .5f) * bin_size_h / roi_bin_grid_h;  // e.g., 0.5, 1.5
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                    const float x = roi_start_w + pw * bin_size_w +
                    (ix + .5f) * bin_size_w / roi_bin_grid_w;

                    float val = bilinear_interpolate(
                        offset_bottom_data, height, width, y, x, index);
                    output_val += val;
                }
            }
            output_val /= count;

            top_data[index] = output_val;
        }
    }

    int ROIAlignForwardLaucher(const float* bottom_data, const float spatial_scale, const int num_rois, const int height, const int width,
                               const int channels, const int aligned_height, const int aligned_width,  const int sampling_ratio,
                               const float* bottom_rois, float* top_data, cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * aligned_height * aligned_width * channels;
        cudaError_t err;


        ROIAlignForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
          output_size, bottom_data, spatial_scale, height, width, channels,
          aligned_height, aligned_width, sampling_ratio, bottom_rois, top_data);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }

    /*** Backward ***/
    inline __device__ float gpu_atomic_add(const float val, float* address);
    inline __device__ float gpu_atomic_add(const float val, float* address) {
        return atomicAdd(address, val);
    }

    __device__ void bilinear_interpolate_gradient(const int height, const int width, float y, float x,
                                                  float& w1, float& w2, float& w3, float& w4,
                                                  int& x_low, int& x_high, int& y_low, int& y_high,
                                                  const int index /* index for debug only*/) {
        // deal with cases that inverse elements are out of feature map boundary
        if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            w1 = w2 = w3 = w4 = 0.;
            x_low = x_high = y_low = y_high = -1;
            return;
        }

        if (y <= 0) {
            y = 0;
        }
        if (x <= 0) {
            x = 0;
        }

        y_low = (int)y;
        x_low = (int)x;

        if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (float)y_low;
        } else {
            y_high = y_low + 1;
        }

        if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (float)x_low;
        } else {
            x_high = x_low + 1;
        }

        float ly = y - y_low;
        float lx = x - x_low;
        float hy = 1. - ly, hx = 1. - lx;

        w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        return;
    }

    __global__ void ROIAlignBackward(const int nthreads, const float* top_diff, const float spatial_scale, const int height, const int width,
                                     const int channels, const int aligned_height, const int aligned_width, const int sampling_ratio,
                                     float* bottom_diff, const float* bottom_rois) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            // (n, c, ph, pw) is an element in the aligned output
            int pw = index % aligned_width;
            int ph = (index / aligned_width) % aligned_height;
            int c  = (index / aligned_width / aligned_height) % channels;
            int n  = index / aligned_width / aligned_height / channels;

            const float* offset_bottom_rois = bottom_rois + n * 5;
            int roi_batch_ind = offset_bottom_rois[0];

            // Do not using rounding; this implementation detail is critical
            float roi_start_w = offset_bottom_rois[1] * spatial_scale;
            float roi_start_h = offset_bottom_rois[2] * spatial_scale;
            float roi_end_w = offset_bottom_rois[3] * spatial_scale;
            float roi_end_h = offset_bottom_rois[4] * spatial_scale;

            // Force malformed ROIs to be 1x1
            float roi_width = fmaxf(roi_end_w - roi_start_w, 1.f);
            float roi_height = fmaxf(roi_end_h - roi_start_h, 1.f);
            float bin_size_h = roi_height / aligned_height;
            float bin_size_w = roi_width / aligned_width;

            float* offset_bottom_diff =
                bottom_diff + (roi_batch_ind * channels + c) * height * width;

            int top_offset = (n * channels + c) * aligned_height * aligned_width;
            const float* offset_top_diff = top_diff + top_offset;
            const float top_diff_this_bin = offset_top_diff[ph * aligned_width + pw];

            // We use roi_bin_grid to sample the grid and mimic integral
            int roi_bin_grid_h = (sampling_ratio > 0)
                ? sampling_ratio
                : ceil(roi_height / aligned_height); // e.g., = 2
            int roi_bin_grid_w =
                (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / aligned_width);

            // We do average (integral) pooling inside a bin
            const float count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

            for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
            {
                const float y = roi_start_h + ph * bin_size_h +
                    (iy + .5f) * bin_size_h / roi_bin_grid_h; // e.g., 0.5, 1.5
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                    const float x = roi_start_w + pw * bin_size_w +
                        (ix + .5f) * bin_size_w / roi_bin_grid_w;

                    float w1, w2, w3, w4;
                    int x_low, x_high, y_low, y_high;

                    bilinear_interpolate_gradient(
                        height, width, y, x, w1, w2, w3, w4,
                        x_low, x_high, y_low, y_high, index);

                    float g1 = top_diff_this_bin * w1 / count;
                    float g2 = top_diff_this_bin * w2 / count;
                    float g3 = top_diff_this_bin * w3 / count;
                    float g4 = top_diff_this_bin * w4 / count;

                    if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
                        // atomicAdd(offset_bottom_diff + y_low * width + x_low, g1);
                        // atomicAdd(offset_bottom_diff + y_low * width + x_high, g2);
                        // atomicAdd(offset_bottom_diff + y_high * width + x_low, g3);
                        // atomicAdd(offset_bottom_diff + y_high * width + x_high, g4);
                        gpu_atomic_add(g1, offset_bottom_diff + y_low * width + x_low);
                        gpu_atomic_add(g2, offset_bottom_diff + y_low * width + x_high);
                        gpu_atomic_add(g3, offset_bottom_diff + y_high * width + x_low);
                        gpu_atomic_add(g4, offset_bottom_diff + y_high * width + x_high);
                    } // if
                } // ix
            } // iy
        } // CUDA_1D_KERNEL_LOOP
    } // RoIAlignBackward

    int ROIAlignBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois, const int height, const int width,
                                const int channels, const int aligned_height, const int aligned_width, const int sampling_ratio,
                                const float* bottom_rois, float* bottom_diff, cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * aligned_height * aligned_width * channels;
        cudaError_t err;

        ROIAlignBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
          output_size, top_diff, spatial_scale, height, width, channels,
          aligned_height, aligned_width,  sampling_ratio, bottom_diff, bottom_rois);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }


#ifdef __cplusplus
}
#endif
