// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"

// Convert byte types for bvecs
void block_convert_bvecs(std::ifstream &reader, std::ofstream &writer, uint8_t *read_buf, uint8_t *write_buf,
                         size_t npts, size_t ndims)
{
    // Read the block of data from input file
    reader.read((char *)read_buf, npts * (ndims * sizeof(uint8_t) + sizeof(uint32_t)));

    // Process each point, skip the first 4 bytes (uint32_t) and copy the vector data
    for (size_t i = 0; i < npts; i++)
    {
        memcpy(write_buf + i * ndims, (read_buf + i * (ndims + sizeof(uint32_t))) + sizeof(uint32_t),
               ndims * sizeof(uint8_t));
    }

    // Write the processed block to output file
    writer.write((char *)write_buf, npts * ndims * sizeof(uint8_t));
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << argv[0] << " <input_bvecs> <output_bin>" << std::endl;
        exit(-1);
    }

    std::ifstream reader(argv[1], std::ios::binary | std::ios::ate);
    if (!reader.is_open())
    {
        std::cerr << "Error opening input file: " << argv[1] << std::endl;
        exit(-1);
    }

    size_t fsize = reader.tellg();
    reader.seekg(0, std::ios::beg);

    uint32_t ndims_u32;
    reader.read((char *)&ndims_u32, sizeof(uint32_t));
    reader.seekg(0, std::ios::beg);
    size_t ndims = (size_t)ndims_u32;
    size_t npts = fsize / ((ndims * sizeof(uint8_t)) + sizeof(uint32_t));
    std::cout << "Dataset: #pts = " << npts << ", #dims = " << ndims << std::endl;

    size_t blk_size = 131072;
    size_t nblks = (npts + blk_size - 1) / blk_size; // Round up npts / blk_size
    std::cout << "# blks: " << nblks << std::endl;

    std::ofstream writer(argv[2], std::ios::binary);
    if (!writer.is_open())
    {
        std::cerr << "Error opening output file: " << argv[2] << std::endl;
        exit(-1);
    }

    int32_t npts_s32 = (int32_t)npts;
    int32_t ndims_s32 = (int32_t)ndims;
    writer.write((char *)&npts_s32, sizeof(int32_t));
    writer.write((char *)&ndims_s32, sizeof(int32_t));

    size_t chunknpts = std::min(npts, blk_size);
    uint8_t *read_buf = new uint8_t[chunknpts * ((ndims * sizeof(uint8_t)) + sizeof(uint32_t))];
    uint8_t *write_buf = new uint8_t[chunknpts * ndims * sizeof(uint8_t)];

    for (size_t i = 0; i < nblks; i++)
    {
        size_t cblk_size = std::min(npts - i * blk_size, blk_size);
        block_convert_bvecs(reader, writer, read_buf, write_buf, cblk_size, ndims);
        std::cout << "Block #" << i << " written" << std::endl;
    }

    delete[] read_buf;
    delete[] write_buf;

    reader.close();
    writer.close();

    return 0;
}
