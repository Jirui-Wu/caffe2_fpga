//https://github.com/DiederikVink/fpga_mm/blob/sw_emu/core/src/data_proc.cpp
//support functions for host.cpp as data_proc.hpp included

#include "caffe2/utils/mesh_processor.hpp"
#include "caffe2/utils/data_proc.hpp"
#include "caffe2/utils/utils.hpp"
#include <cstdlib>
#include <limits>

namespace caffe2 {
namespace math {

void clean_string(std::string& word) {
//{{{
    auto i = 0;
    while(i < word.size()) {
        if (word[i] == '[' || word[i] == ']') word.erase(i,1);
        else i++;
    }
} //}}}

void add_list(std::stringstream& listData, std::vector<unsigned>& targetVec) {
//{{{
    for (std::string value; std::getline(listData, value, ',');) {
        clean_string(value);
        if (value != "") targetVec.push_back(std::stoi(value));
    }
} //}}}

// load layer stats
void load_stats(std::vector<std::string>& layers, std::map< std::string,std::vector<unsigned> >& stats, const std::string& fileName) {
//{{{
    std::ifstream statsFile(fileName + "sizes.csv");
    std::string line;
    int count = 0;
    int layer = 0;
	while (std::getline(statsFile, line))
	{
        if (count > 0) {
            std::string value;
            std::stringstream ss(line);
            std::vector<std::string> sizeVec;
            while (std::getline(ss, value, '|')) {
                sizeVec.push_back(value);
            }

            std::stringstream stride(sizeVec[1]);
            std::stringstream pad(sizeVec[2]);
            std::stringstream kernel(sizeVec[3]);
            std::stringstream input(sizeVec[4]);
            std::stringstream im2col(sizeVec[5]);
            std::stringstream output(sizeVec[6]);

            layers.push_back(sizeVec[0]);

            add_list(kernel, stats[sizeVec[0] + "-kernel"]);
            add_list(input, stats[sizeVec[0] + "-input"]);
            add_list(im2col, stats[sizeVec[0] + "-i2c"]);
            add_list(output, stats[sizeVec[0] + "-output"]);
            stats[sizeVec[0] + "-stride"].push_back(std::stoi(stride.str()));
            stats[sizeVec[0] + "-pad"].push_back(std::stoi(pad.str()));
        }
        count++;
	}
	statsFile.close();
}
//}}}

// load input feature matrix (IFM)
void load_ifm(ifm_t* ifmMat, const std::string& fileName, const int batchSize)
//{{{
{
    std::cout << "Loading IFM...";

    int size = 0;
    for (int i = 0; i < batchSize; i++) {
        std::vector<ifm_t> inputs;
        load_data(inputs, fileName + "layer-in" + std::to_string(i) + ".csv");
	    for (int elementIdx = 0; elementIdx < inputs.size(); elementIdx++) { ifmMat[elementIdx+(inputs.size()*i)] = inputs[elementIdx]; }
        size += inputs.size();
    }
	std::cout << "Loaded " << size << " elements" << std::endl;
}
//}}}

// load weights matrix
void load_weights(w_t* weightsMat, const std::string& fileName)
//{{{
{
    std::cout << "Loading Weights...";
	std::vector<w_t> weights;
    load_data(weights, fileName + "weight.csv");
	for (int weightIdx = 0; weightIdx < weights.size(); weightIdx++)
	{
		weightsMat[weightIdx] = weights[weightIdx];
	}
	std::cout << "Loaded " << weights.size() << " elements" << std::endl;
}
//}}}

// laod biases
void load_bias(ofm_t* biasVec)
//{{{
{
    std::cout << "Loading Biases...";
	std::vector<ofm_t> biases;
    load_data(biases, "data/bias_layer1_raw.txt");

    for (int biasIdx = 0; biasIdx < biases.size(); biasIdx++)
    {
        biasVec[biasIdx] = biases[biasIdx];
    }
}
//}}}

void exp_out(const w_t* weightsMat, const ifm_t* ifmMat, std::vector<ofm_t>& expOfm, const int (&params)[11])
//{{{
{
    for (auto batch = 0; batch < params[0]; ++batch) {
        for (auto ifmh = 0; ifmh < params[1]; ++ifmh) {
            for (auto ifmw = 0; ifmw < params[2]; ++ifmw) {
                for( auto ifmc = 0; ifmc < params[3]; ++ifmc) {
                    expOfm[batch*params[1] + ifmh*params[2] + ifmw*params[3] + ifmc] = 0;
                }
            }
        }
    }
}
//}}}

// // calc and write OFM data
// void cpu_calc_write_ofm(std::vector<ofm_t, aligned_allocator<ofm_t>> &ofmExp,
//         const std::vector<ifm_t, aligned_allocator<ifm_t>> ifmVec,
//         const std::vector<ifm_t, aligned_allocator<w_t>> weightsVec,
//         const int* params)
// //{{{
// {
//     int aTile, bTile, cTile;
//     // for (int tile1 = 0; tile1 < weightsParams[0]; tile1++)
//     for (int tile1 = 0; tile1 < params[0]; tile1++)
//     {
//         // bTile = 0;
//         for (int tile2 = 0; tile2 < params[1]; tile2++)
//         {
//             // aTile = tile1*weightsParams[1];
//             for (int t = params[3]; t < params[2]; t++)
//             {
//                 // int aTile = tile1 * weightsParams[1] + t;
//                 // int bTile = tile2 * weightsParams[1] + t;
//                 aTile = tile1 * params[1] + t;
//                 bTile = tile2 * params[1] + t;
//                 for (int i=0; i<TILE_ROW; i++)
//                 {
//                     for (int j=0; j<TILE_COL; j++)
//                     {
//                         for (int k=0; k<TILE_COMMON; k++)
//                         {
//                             cTile = tile1*params[1] + tile2;
//                             ofmExp[cTile*TILE_ROW*TILE_COL + i*TILE_COL + j] += weightsVec[aTile*TILE_ROW*TILE_COMMON + i*TILE_COMMON + k] * ifmVec[bTile*TILE_COL*TILE_COMMON + k*TILE_COL + j];
//                         }
//                     }
//                 }
//                 // aTile ++;
//                 // bTile ++;
//             }
//         }
//     }
//
//     static ofm_t ofmMat[C_ROW * C_COL];
//     TransformToMatrixLayoutFunc(
//         ofmExp,
//         ofmMat,
//         TILE_ROW,
//         TILE_COL,
//         A_ROW,
//         B_COL,
//         false
//     );
//
//     std::ofstream tmpFileOFM("data/tmpOFM.txt");
//     for (int i=0; i<C_ROW; i++)
//     {
//         for (int j=0; j<C_COL; j++)
//         {
//             tmpFileOFM << ofmMat[i*C_COL + j] << " " ;
//         }
//         tmpFileOFM << std::endl;
//     }
// }
// //}}}

int verify_result()
{{{
	std::vector<w_t> cpuElements;
    std::vector<w_t> fpgaElements;
    float inputElement;

    std::ifstream cpuFile("data/tmpOFM.txt");
	while (cpuFile >> inputElement)
	{
		cpuElements.push_back(inputElement);
	}
	cpuFile.close();

    std::ifstream fpgaFile("data/tmpTile.txt");
	while (fpgaFile >> inputElement)
	{
		fpgaElements.push_back(inputElement);
	}
    fpgaFile.close();

    std::cout << "App finished! " << fpgaElements.size() << std::endl;
    auto success = true;
    for (auto idx = 0; idx < fpgaElements.size(); idx++)
    {
        if (fpgaElements[idx] != cpuElements[idx])
        {
            std::cout << "Index: " << idx << " FPGA: " << fpgaElements[idx] << " CPU: " << cpuElements[idx] << std::endl;
            success = false;
        }
    }

    if (success) std::cout << "Success!" << std::endl << std::endl;
    else std::cout << "Mismatched Result!" << std::endl << std::endl;

    return success;
}}}

int verify_result(std::vector<ofm_t, aligned_allocator<ofm_t>>& fpgaVec)
{{{
    // float inputElement;
    std::string inputLine;
    std::string inputElement;
	std::vector<ofm_t> cpuVec;
    auto org = 0;
    std::string targetFile;
    if (org == 0) {
        targetFile = "data/output.csv";
    }
    else {
        targetFile = "data/output_og.csv";
    }
    std::ifstream cpuFile(targetFile);

    if (cpuFile.is_open()) {
        while(std::getline(cpuFile, inputLine)) {
            std::stringstream ssLine(inputLine);

            while (std::getline(ssLine, inputElement, ',')) {
                cpuVec.push_back(std::stof(inputElement));
            }
        }
    }
	cpuFile.close();

    std::cout << "App finished! " << cpuVec.size() << " " << fpgaVec.size() << std::endl;
    auto success = true;
    // if (fpgaVec.size() != cpuVec.size()) {
    //     std::cout << "FPGA output size is incorrect" << std::endl;
    //     success = false;
    // }

    auto count = 0;
    auto count1 = 0;
    auto count2 = 0;
    auto count3 = 0;
    if (success) {
        std::vector<float> err;
        float me = 0;
        float mse = 0;
        float mes = 0;

        int start = 0;
        int end = fpgaVec.size();

        for (auto idx = start; idx < end; idx++) {
            // err.push_back(abs(cpuVec[idx] - fpgaVec[idx])/cpuVec[idx]);
            err.push_back(abs(cpuVec[idx] - fpgaVec[idx]));
        }

        if (org == -1) {
            for (auto idx = start; idx < end; idx++) {
                if (idx % 96 == 0) {
                    count += 1;
                }
                if (idx % 16 == 0) {
                    std::cout << "===========" << count << "===========" << std::endl;
                }
                std::cout << cpuVec[idx] << " " << fpgaVec[idx] << " " << cpuVec[idx] - fpgaVec[idx] << std::endl;
            }
        }

        end = start + fpgaVec.size();
        for (auto idx = start; idx < end; idx++) {
            me += err[idx];
            mse += err[idx]*err[idx];
        }
        std::cout << "full val: " << me << " max: " << std::numeric_limits<float>::max() << std::endl;
        me /= (end-start);
        mes = me * me;
        mse /= (end-start);
        std::cout << "a-data\t" << " me: " << me << " mse: " << mse << " mes: " << mes << std::endl;
        me = mse = mes = 0;

        if (org == 0) { end = start + 4*4*6; }
        else { end = start + 112*112*64; }
        for (auto idx = start; idx < end; idx++) {
            me += err[idx];
            mse += err[idx]*err[idx];
        }
        me /= (end-start);
        mes = me * me;
        mse /= (end-start);
        std::cout << "a-filt\t" << " me: " << me << " mse: " << mse << " mes: " << mes << std::endl;
        me = mse = mes = 0;

        if (org == 0) { end = start + 4*4; }
        else { end = start + 112*112; }
        for (auto idx = start; idx < end; idx++) {
            me += err[idx];
            mse += err[idx]*err[idx];
        }
        me /= (end-start);
        mes = me * me;
        mse /= (end-start);
        std::cout << "1-img\t" << " me: " << me << " mse: " << mse << " mes: " << mes << std::endl;
        me = mse = mes = 0;

        if (org == 0) { end = start + 4; }
        else { end = start + 112; }
        for (auto idx = start; idx < end; idx++) {
            // std::cout << fpgaVec[idx] << " " << cpuVec[idx] << " " << cpuVec[idx] - fpgaVec[idx] << " " << err[idx] << std::endl;
            me += err[idx];
            mse += err[idx]*err[idx];
        }
        me /= (end-start);
        mes = me * me;
        mse /= (end-start);
        std::cout << "1-row\t" << " me: " << me << " mse: " << mse << " mes: " << mes << std::endl;

        // for (auto tmp = 31; tmp < 33; tmp++) {
        //     start = 112*112*tmp;
        //     int row = 0;
        //     for (auto idx = start; idx < start+12544; idx += 112) {
        //         end = idx + 112;
        //         for (auto idx = start; idx < end; idx++) {
        //             me += err[idx];
        //             mse += err[idx]*err[idx];
        //         }
        //         me /= (end-start);
        //         mes = me * me;
        //         mse /= (end-start);
        //         std::cout << idx << "-row\t" << " me: " << me << " mse: " << mse << " mes: " << mes << std::endl;
        //     }
        //     std::cout << "------------------" << std::endl;
        // }
        // std::ofstream resFile("result.csv");
        // if (resFile.is_open()) {
        //     int count;
        //     for (auto val: fpgaVec) {
        //         resFile << val << ", ";
        //         count++;
        //         if (count == 112*112*64) { resFile << std::endl; count = 0;}
        //     }
        // }
	    // resFile.close();

        // start = 112;
        // for (auto idx = start; idx < start+112; idx++) {
        //     std::cout << "f-c: " << fpgaVec[idx] << " - " << cpuVec[idx] << " = " << fpgaVec[idx] - cpuVec[idx] << std::endl;
        // }




    }
    // std::ofstream tmpFileOFM("data/tmpOFM.txt");
    // for (auto idx = 0; idx < fpgaVec.size(); ++idx) {
    //     // std::cout << fpgaVec[idx] << " " << std::endl;
    //     tmpFileOFM << fpgaVec[idx] << " ";
    // }
    // tmpFileOFM << std::endl;


    if (success) std::cout << "Success!" << std::endl << std::endl;
    else std::cout << "Mismatched Result!" << std::endl << std::endl;

    return success;
}}}

template <typename matType>
void load_data(std::vector<matType>& inputElements, const std::string fileName)
{{{
	std::ifstream matFile(fileName);
	matType inputElement;
    std::string comma;
	while (matFile >> inputElement >> comma)
	{
		inputElements.push_back(inputElement);
	}
	matFile.close();
    assert (inputElements.size() != 0);
}}}

template <typename matType>
void save_data(const std::vector<matType, aligned_allocator<matType>> vec, const int* params, const std::string fileName)
{{{
    std::ofstream tmpFile(fileName);
    int flatIdx = 0;
    for (int k=0; k<params[1]; k++)
    {
	    for (int i=0; i<params[2]; i++)
	    {
	    	for (int j=0; j<params[3]; j++)
	    	{
	    		tmpFile << vec[flatIdx] << " ";
                flatIdx ++;
	    	}
	    	tmpFile << std::endl;
	    }
        tmpFile << std::endl;
        tmpFile << std::endl;
    }
}}}

union cast_t
{
    public:
        float f;
        unsigned u;
};

template <typename matType>
void create_mat(std::string seedVal, matType* mat, const long matSize, const int batchSize, const int value)
{{{

    // std::ofstream outFile("input.csv", std::ios::out);
    // if (outFile.is_open()) {
    std::seed_seq seed(seedVal.begin(),seedVal.end());
    if (value == 0) {
        for (auto batch = 0; batch < batchSize; ++batch) {
            for (auto elem = 0; elem < matSize; ++elem) {
                mat[batch*matSize + elem] = 0;
            }
        }
    }
    else
    {
        std::default_random_engine generator(seed);
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (auto batch = 0; batch < batchSize; ++batch) {
            for (auto elem = 0; elem < matSize; ++elem) {
                cast_t val;
                val.f = distribution(generator);
                mat[batch*matSize + elem] = val.f;
                // outFile << mat[batch*matSize + elem] << ", ";
            }
            // outFile << std::endl;
        }
    }
    std::cout << "Created matrix...";
    // outFile.close();
    // }
}}}
template void create_mat<w_t> (std::string, w_t*, const long, const int, const int);
}//math
}//caffe2
