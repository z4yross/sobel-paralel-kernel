#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <iostream>

#include <sys/time.h>

// #include <pthread.h>

// #include <omp.h>

using namespace cv;

struct argsSobel
{
   Mat *img;
   Mat *out;
   int idxI;
   int idxF;
   int th;
};

struct argsBlur
{
   Mat *img;
   Mat *out;
   int idxI;
   int idxF;
   int blurSize;
   double **kernel;
};

std::string name;

void gaussKernel(int size, double sigma, int K, double * out)
{
   double sum = 0;
   int i, j;

   for (int i = 0; i < size; i++)
   {
      for (j = 0; j < size; j++)
      {
         double x = i - (size - 1) / 2.0;
         double y = j - (size - 1) / 2.0;
         out[i * size + j] = K * exp(((pow(x, 2) + pow(y, 2)) / ((2 * pow(sigma, 2)))) * (-1));
         sum += out[i * size + j];
      }
   }

   for (i = 0; i < size; i++)
      for (j = 0; j < size; j++)
         out[i * size + j] /= sum;

}

__global__ void gaussBlur(double* kernel, int blurSize, uchar* img, int img_step, int img_cols, int img_rows, uchar* out)
{
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int idx = index; idx < img_cols * img_rows; idx += stride)
   {
      int i = idx % img_cols;
      int j = idx / img_cols;

      float sum = 0.0;

      for (int k = 0; k < blurSize; k++)
      {
         for (int l = 0; l < blurSize; l++)
         {
            int x = i + (k - blurSize / 2);
            int y = j + (l - blurSize / 2);

            if (x >= 0 && x < img_cols && y >= 0 && y < img_rows)
            {
               double kV = kernel[k * blurSize + l];
               sum += img[y * img_cols + x] * kV;
            }
         }
      }

      out[idx] = sum;
   }
}

__global__ void sobel(int treshold, uchar* img, int img_step, int img_cols, int img_rows, uchar* out)
{

   int kS = 3;
   double sobelX[3][3] = {
       {1.0, 0.0, -1.0},
       {2.0, 0.0, -2.0},
       {1.0, 0.0, -1.0}};

   double sobelY[3][3] = {
       {1.0, 2.0, 1.0},
       {0.0, 0.0, 0.0},
       {-1.0, -2.0, -1.0}};

   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int idx = index; idx < img_cols * img_rows; idx += stride)
   {
      int i = idx % img_cols;
      int j = idx / img_cols;

      float sumX = 0.0;
      float sumY = 0.0;

      for (int k = 0; k < kS; k++)
      {
         for (int l = 0; l < kS; l++)
         {
            int x = i + (k - kS / 2);
            int y = j + (l - kS / 2);

            if (x >= 0 && x < img_cols && y >= 0 && y < img_rows)
            {
               sumX += sobelX[k][l] * img[y * img_cols + x];
               sumY += sobelY[k][l] * img[y * img_cols + x];
            }
         }
      }

      int v = (int)(sqrt(sumX * sumX + sumY * sumY) / 1448 * 256);
      int a = max(v, treshold);

      if (a == treshold) out[idx] = 0;
      else out[idx] = 255;
   }
}

void bW(Mat *img)
{
   int rows = ((Mat)*img).rows;
   int cols = ((Mat)*img).cols;

   for (int i = 0; i < rows; i++)
   {
      for (int j = 0; j < cols; j++)
      {
         Vec3b p = ((Mat)*img).at<Vec3b>(i, j);
         unsigned char gray = p[0] * 0.3 + p[1] * 0.58 + p[2] * 0.11;
         ((Mat)*img).at<Vec3b>(i, j) = Vec3b(gray, gray, gray);
      }
   }
}

float sobelSq(std::string iName, int th, bool blr, int bS, int bSg, int blockSize, int blocks)
{
   name = iName;

   struct timeval tval_before, tval_after, tval_result;
   gettimeofday(&tval_before, NULL);

   std::string image_path = samples::findFile("images/" + name);
   Mat imgFC = imread(image_path, IMREAD_COLOR);
   Mat img;

   cvtColor(imgFC, img, cv::COLOR_BGRA2GRAY);

   
   uchar* p_img;
   uchar* p_out;
   cudaMallocManaged(&p_img, sizeof(uchar) * img.rows * img.cols);
   cudaMallocManaged(&p_out, sizeof(uchar) * img.rows * img.cols);

   Mat in(img.rows, img.cols, img.type(), p_img);
   Mat out(img.rows, img.cols, img.type(), p_out);

   cuda::GpuMat gpuImg(img.rows, img.cols, img.type(), p_img);
   cuda::GpuMat gpuOut(img.rows, img.cols, img.type(), p_out);

   img.copyTo(in);
   gpuImg.upload(img);

   img.release();

   double *kernel;
   cudaMallocManaged(&kernel, sizeof(double) * bS * bS );

   // ----------------- BLUR-----------------
   if (blr)
   {
      gaussKernel(bS, bSg, 1, kernel);     

      gaussBlur<<<blocks, blockSize>>>((double *) kernel, bS, (uchar *) p_img, gpuImg.step, gpuImg.cols, gpuImg.rows, (uchar *) p_out);
      cudaDeviceSynchronize();

      cudaError_t error = cudaGetLastError();
      if (error != cudaSuccess) {
         fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
      }
   }

   out.copyTo(in);
   
   // ----------------- SOBEL-----------------

   sobel<<<blocks, blockSize>>>(th, (uchar *) p_img, gpuImg.step, gpuImg.cols, gpuImg.rows, (uchar *) p_out);
   cudaDeviceSynchronize();

   cudaError_t error = cudaGetLastError();
   if (error != cudaSuccess) {
      fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
   }

   gettimeofday(&tval_after, NULL);
   timersub(&tval_after, &tval_before, &tval_result);
   // printf("%d threads -> %ld.%06lds\n", p, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

   imwrite("out/" + name, out);

   // free(prms);
   out.release();
   in.release();

   gpuImg.release();
   gpuOut.release();

   cudaFree(p_img);
   cudaFree(p_out);
   cudaFree(kernel);

   // String res = tval_result.tv_sec + "." + tval_result.tv_usec;
   // std::cout << res << std::endl;
   return (long int)tval_result.tv_sec + (long int)tval_result.tv_usec * 0.000001;
}