#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "npy.hpp"
#include "kernels.cu"
#include <chrono>

using namespace std;


void load_data(
    std::string data_dir,
    int n, int w, int h,
    float *depths,
    float *normals,
    float *intr_matrices,
    float *extr_matrices,
    float *intr_matrices_inv,
    float *extr_matrices_inv
) {

    std::vector<unsigned long> shape{};
    bool fortran_order;
    std::vector<float> data;

    auto start_time = std::chrono::steady_clock::now();

    for (int i = 0; i < n; i++) {
        string depth_path = data_dir + "/depths/" + to_string(i + 1) + ".npy";
        npy::LoadArrayFromNumpy(depth_path, shape, fortran_order, data);
        copy(data.data(), data.data() + h * w, depths + i * h * w);

        string normal_path = data_dir + "/normals/" + to_string(i + 1) + ".npy";
        npy::LoadArrayFromNumpy(normal_path, shape, fortran_order, data);
        copy(data.data(), data.data() + h * w * 3, normals + i * h * w * 3);
    }


    string intr_path = data_dir + "/intrinsic.npy";
    npy::LoadArrayFromNumpy(intr_path, shape, fortran_order, data);
    copy(data.data(), data.data() + 9 * n, intr_matrices);

    string extr_path = data_dir + "/extrinsic.npy";
    npy::LoadArrayFromNumpy(extr_path, shape, fortran_order, data);
    copy(data.data(), data.data() + 16 * n, extr_matrices);

    string intr_inv_path = data_dir + "/intrinsic_inv.npy";
    npy::LoadArrayFromNumpy(intr_inv_path, shape, fortran_order, data);
    copy(data.data(), data.data() + 9 * n, intr_matrices_inv);

    string extr_inv_path = data_dir + "/extrinsic_inv.npy";
    npy::LoadArrayFromNumpy(extr_inv_path, shape, fortran_order, data);
    copy(data.data(), data.data() + 16 * n, extr_matrices_inv);


    auto end_time = std::chrono::steady_clock::now();
    
    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    ofstream myfile;
    string filename = "output/" + std::to_string(n) + "_times.txt";
    myfile.open (filename, std::ios_base::app);
    myfile << "Data Time = " << seconds << " seconds\n";
    myfile.close();
}


__global__ void image2world(
    int n, int w, int h, 
    float *xyzs, float *depths, 
    float *intr_mats_inv, float *extr_mats
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    for (int img = 0; img < n; img++) {
        float depth = depths[img * w * h + i * w + j];
        
        float *xyz = xyzs + 3 * (img * w * h + i * w + j); 
        float *intr_inv = intr_mats_inv + img * 9;
        float *extr = extr_mats + img * 16;

        float3 cam = {
            j * depth * intr_inv[0] + i * depth * intr_inv[1] + depth * intr_inv[2],
            j * depth * intr_inv[3] + i * depth * intr_inv[4] + depth * intr_inv[5],
            j * depth * intr_inv[6] + i * depth * intr_inv[7] + depth * intr_inv[8],
        };

        xyz[0] = extr[0] * cam.x + extr[1] * cam.y + extr[2] * cam.z + extr[3];
        xyz[1] = extr[4] * cam.x + extr[5] * cam.y + extr[6] * cam.z + extr[7];
        xyz[2] = extr[8] * cam.x + extr[9] * cam.y + extr[10] * cam.z + extr[11];
    }
}



__global__ void compute_irradiances(
    int n, int w, int h,
    float *xyzs, float *depths, float *normals,
    float *intr_mats, float *extr_mats,
    float *intr_mats_inv, float *extr_mats_inv,
    float *total_irradiance
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    for (int cam_i = 0; cam_i < n; cam_i++) {
        for (int cam_j = 0; cam_j < n; cam_j++) {
            // world to camera
            float *xyz_i = xyzs + 3 * (cam_i * w * h + i * w + j);
            float *extr_inv_j = extr_mats_inv + cam_j * 16;
            float *intr_j = intr_mats + cam_j * 9;

            float3 cam = {
                extr_inv_j[0] * xyz_i[0] + extr_inv_j[1] * xyz_i[1] + extr_inv_j[2] * xyz_i[2] + extr_inv_j[3],
                extr_inv_j[4] * xyz_i[0] + extr_inv_j[5] * xyz_i[1] + extr_inv_j[6] * xyz_i[2] + extr_inv_j[7],
                extr_inv_j[8] * xyz_i[0] + extr_inv_j[9] * xyz_i[1] + extr_inv_j[10] * xyz_i[2] + extr_inv_j[11]
            };

            float3 xyd = {
                intr_j[3] * cam.x + intr_j[4] * cam.y + intr_j[5] * cam.z,
                intr_j[0] * cam.x + intr_j[1] * cam.y + intr_j[2] * cam.z,
                intr_j[6] * cam.x + intr_j[7] * cam.y + intr_j[8] * cam.z
            };

            int x_j = (int) (xyd.x / xyd.z) ;
            int y_j = (int) (xyd.y / xyd.z);
            float d_j = xyd.z;
            float depth_j = -1.;

            if (x_j >= 0 && x_j < h && y_j >=0 && y_j < w) {
                depth_j = depths[cam_j * w * h + x_j * w + y_j];
            }

            float *normal_i = normals + 3 * (cam_i * w * h + i * w + j);
            float3 displacement_other = {
                xyz_i[0] - extr_mats[cam_j * 16 + 3],
                xyz_i[1] - extr_mats[cam_j * 16 + 7],
                xyz_i[2] - extr_mats[cam_j * 16 + 11]
            };

            float direction_norm_sqr = 
                displacement_other.x * displacement_other.x + 
                displacement_other.y * displacement_other.y + 
                displacement_other.z * displacement_other.z;
            
            float direction_norm = sqrt(direction_norm_sqr);

            float3 neg_direction_other = {
                displacement_other.x / -direction_norm,
                displacement_other.y / -direction_norm,
                displacement_other.z / -direction_norm,
            };

            float cosine_law = 
                neg_direction_other.x * normal_i[0] + 
                neg_direction_other.y * normal_i[1] +
                neg_direction_other.z * normal_i[2];

            if (depth_j > d_j-0.01 && d_j > 0. && cosine_law > 0)
                total_irradiance[cam_i * w * h + i * w + j] += cosine_law * (1. / direction_norm_sqr);
        }
    }
}

int main(int argc, char **argv)
{
    printf("Running %d frames...\n", stoi(argv[1]));

    // You can directly tune n for profiling
    // but h and w aren't changable unless 
    // you make changes to the data as well
    // int n = 100;
    int n = stoi(argv[1]);
    int h = 1024;
    int w = 1024;

    float *depths_cpu = (float *) malloc(sizeof(float) * n * w * h);
    float *normals_cpu = (float *) malloc(sizeof(float) * n * w * h * 3);
    float *xyzs_cpu = (float *) malloc(sizeof(float) * n * w * h * 3);
    float *intr_matrices_cpu = (float *) malloc(sizeof(float) * n * 9);
    float *extr_matrices_cpu = (float *) malloc(sizeof(float) * n * 16);
    float *intr_matrices_inv_cpu = (float *) malloc(sizeof(float) * n * 9);
    float *extr_matrices_inv_cpu = (float *) malloc(sizeof(float) * n * 16);
    float *total_irradiance_cpu = (float *) malloc(sizeof(float) * n * w * h);

    string path{"/home/haodali/cs267_final_project/dataset_100"};
    load_data(path, n, w, h, 
        depths_cpu, normals_cpu, 
        intr_matrices_cpu, extr_matrices_cpu, 
        intr_matrices_inv_cpu, extr_matrices_inv_cpu
    );
    

    float *depths;
    cudaMalloc((void**) &depths, sizeof(float) * n * w * h);
    cudaMemcpy(depths, depths_cpu, sizeof(float) * n * w * h, cudaMemcpyHostToDevice);
    
    float *normals;
    cudaMalloc((void**) &normals, sizeof(float) * n * w * h * 3);
    cudaMemcpy(normals, normals_cpu, sizeof(float) * n * w * h * 3, cudaMemcpyHostToDevice);

    float *xyzs;
    cudaMalloc((void**) &xyzs, sizeof(float) * n * w * h * 3);
    
    float *intr_matrices;
    cudaMalloc((void**) &intr_matrices, sizeof(float) * n * 9);
    cudaMemcpy(intr_matrices, intr_matrices_cpu, sizeof(float) * n * 9, cudaMemcpyHostToDevice);
    
    float *extr_matrices;
    cudaMalloc((void**) &extr_matrices, sizeof(float) * n * 16);
    cudaMemcpy(extr_matrices, extr_matrices_cpu, sizeof(float) * n * 16, cudaMemcpyHostToDevice);
    
    float *intr_matrices_inv;
    cudaMalloc((void**) &intr_matrices_inv, sizeof(float) * n * 9);
    cudaMemcpy(intr_matrices_inv, intr_matrices_inv_cpu, sizeof(float) * n * 9, cudaMemcpyHostToDevice);
    
    float *extr_matrices_inv; 
    cudaMalloc((void**) &extr_matrices_inv, sizeof(float) * n * 16);
    cudaMemcpy(extr_matrices_inv, extr_matrices_inv_cpu, sizeof(float) * n * 16, cudaMemcpyHostToDevice);

    float *total_irradiance;
    cudaMalloc((void **)&total_irradiance, sizeof(float) * n * w * h);
    cudaMemset(total_irradiance, 0., n * w * h);

    dim3 n_blocks(32, 32);
    dim3 n_threads(32, 32);
    
    auto start_time = std::chrono::steady_clock::now();
    image2world<<<n_blocks, n_threads>>>(n, w, h, xyzs, depths, intr_matrices_inv, extr_matrices);
    cudaDeviceSynchronize();

    compute_irradiances<<<n_blocks, n_threads>>>(
        n, w, h,
        xyzs, depths, normals, 
        intr_matrices, extr_matrices,
        intr_matrices_inv, extr_matrices_inv,
        total_irradiance
    );
    cudaDeviceSynchronize();

    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double seconds = diff.count();

    ofstream myfile;
    string filename = "output/" + std::to_string(n) + "_times.txt";
    myfile.open (filename, std::ios_base::app);
    myfile << "Simulation Time = " << seconds << " seconds\n";
    myfile.close();

    cudaMemcpy(total_irradiance_cpu, total_irradiance, sizeof(float) * n * w * h, cudaMemcpyDeviceToHost);
    cudaMemcpy(xyzs_cpu, xyzs, sizeof(float) * n * w * h * 3, cudaMemcpyDeviceToHost);

    const std::vector<size_t> shape{n, h, w};
    npy::SaveArrayAsNumpy("irr.npy", false, shape.size(), shape.data(), total_irradiance_cpu);
    
}
