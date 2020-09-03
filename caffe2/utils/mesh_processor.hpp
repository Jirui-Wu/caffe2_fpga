#ifndef MESH_PROCESSOR_HPP
#define MESH_PROCESSOR_HPP

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <assert.h>
#include <ap_int.h>
/* --- precisions --- */
// typedef int ifm_t;
// typedef int w_t;
// typedef int ofm_t;
// typedef int i2c_t;
// #define _batchSize 128
// #define _D 64

typedef float ifm_t;
typedef float w_t;
typedef float ofm_t;
typedef float i2c_t;

// #define _X 4
// #define _Y 2
// #define _batchSize 128
// #define _D 64
//
// #define _ifmH 224
// #define _ifmW 224
// #define _channels 3
// #define _kern 7
// #define _ofmH 112
// #define _ofmW 112
// #define _ofmC _D
// #define _i2cH 147
// #define _i2cW 12544
// #define _stride 2
// #define _pad 3

typedef float ifm_t;
typedef float w_t;
typedef float ofm_t;
typedef float i2c_t;

#define _X 4
#define _Y 2
#define _batchSize 4
#define _D 2

#define _ifmH 8
#define _ifmW 8
#define _channels 3
#define _kern 4
#define _ofmH 4
#define _ofmW 4
#define _ofmC _D
#define _i2cH 48
#define _i2cW 16
#define _stride 2
//#define _pad 1

#define _interMul 11
#define _interAdd 8
#define _buffer 2
#define _AS1 5
#define _AS2 3
#define _AS 7

#define TARGET_WIDTH 32
#define PORT_WIDTH 512

#define _k2 _kern * _kern
#define _k2c _k2 * _channels

#endif
