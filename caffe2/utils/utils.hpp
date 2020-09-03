#ifndef UTILS_FLOAT_HPP
#define UTILS_FLOAT_HPP

#include <vector>
#include "caffe2/fpga/xcl2.hpp"
#include "math.h"
#include "caffe2/fpga/mesh_processor.hpp"


bool is_a_ge_zero_and_a_lt_b(int a, int b);

void TransformToFlattenTiledLayout(
    const w_t *inputMat, std::vector<w_t, aligned_allocator<w_t>> &tiledFlatMat,
    int* params, int ROWS, int COLS, int tR, int tC, bool transposeTiles,
    bool transposeMat
);

void TransformToMatrixLayoutFunc(
		std::vector<ofm_t, aligned_allocator<ofm_t>> &tiledFlatMat, ofm_t *outputMat,
        int TR, int TC, int ROWS, int COLS, bool transposed
);

void SetupFPGATiling(
    const float *inputMat, int ROWS, int COLS, int tR, int tC,
    bool transposeTiles, bool transposeMat, int * params
);

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, Dtype* data_col
	)
//{{{
{
	const int output_h = (height + 2 * pad_h -
						 (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width + 2 * pad_w -
						 (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	const int channel_size = height * width;
	for (int channel = channels; channel--; data_im += channel_size)
	{
		for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++)
		{
			for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++)
			{
				int input_row = -pad_h + kernel_row * dilation_h;
				for (int output_rows = output_h; output_rows; output_rows--)
				{
					if (!is_a_ge_zero_and_a_lt_b(input_row, height))
					{
						for (int output_cols = output_w; output_cols; output_cols--)
						{
							*(data_col++) = 0;
						}
					}
					else
					{
						int input_col = -pad_w + kernel_col * dilation_w;
						for (int output_col = output_w; output_col; output_col--)
						{
							if (is_a_ge_zero_and_a_lt_b(input_col, width))
							{
								*(data_col++) = data_im[input_row * width + input_col];
							}
							else
							{
								*(data_col++) = 0;
							}
							input_col += stride_w;
						}
					}
					input_row += stride_h;
				}
			}
		}
	}
}
//}}}

#endif
