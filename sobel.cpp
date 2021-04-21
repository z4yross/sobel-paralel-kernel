#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <iostream>

using namespace cv;

std::string name;
int treshold;

bool blur;
int blurSize;
int blurSigma;

double** gaussKernel(int size, double sigma, int K){
   double** gauss = 0;
   gauss = new double*[size];

   double sum = 0;
   int i, j;

   for (i = 0; i < size; i++) {
      gauss[i] = new double[size];
      for (j = 0; j < size; j++) {
         double x = i - (size - 1) / 2.0;
         double y = j - (size - 1) / 2.0;
         gauss[i][j] = K * exp(((pow(x, 2) + pow(y, 2)) / ((2 * pow(sigma, 2)))) * (-1));
         sum += gauss[i][j];
      }
   }
   for (i = 0; i < size; i++) {
      for (j = 0; j < size; j++) {
         gauss[i][j] /= sum;
      }
   }

   return gauss;
}

void gaussBlur(Mat *img){
   
   int kS = 3;
   int gSigma = blurSigma;

   double** kernel = gaussKernel(kS, gSigma, 1);

   int imgW = ((Mat) *img).rows;
   int imgH = ((Mat) *img).cols;

   Mat nx = ((Mat) *img).clone();

   for(int i = 0; i < imgW; i++){
      for(int j = 0; j < imgH; j++){

         float sum = 0.0;

         for(int k = 0; k < kS; k++){
               for(int l = 0; l < kS; l++){
                  int x = i + (k - kS / 2);
                  int y = j + (l - kS / 2);          
                  
               //   std::cout << x << " " << y << " --- "<< i << " " << j << std::endl;

                  if(x >= 0 && x < imgW && y >= 0 && y < imgH){
                     double kV = kernel[k][l];
                     sum += ((Mat) *img).at<Vec3b>(x, y)[0] * kV;
                  } 
               }
         }

         nx.at<Vec3b>(i, j) = Vec3b(sum,sum,sum);
      }
   }

   *img = nx;
}

void sobel(Mat *img){
   int kS = 3;

   double sobelX[3][3] = {
      {1.0, 0.0, -1.0},
      {2.0, 0.0, -2.0},
      {1.0, 0.0, -1.0}
   };

   double sobelY[3][3] = {
      {1.0, 2.0, 1.0},
      {0.0, 0.0, 0.0},
      {-1.0, -2.0, -1.0}
   };

   int imgW = ((Mat) *img).rows;
   int imgH = ((Mat) *img).cols;

   double mx = 0;

   Mat nx = ((Mat) *img).clone();

   for(int i = 0; i < imgW; i++){
      for(int j = 0; j < imgH; j++){

         double sumX = 0.0;
         double sumY = 0.0;

         for(int k = 0; k < kS; k++){
               for(int l = 0; l < kS; l++){
                  int x = i + (k - kS / 2);
                  int y = j + (l - kS / 2);          
                  
               //   std::cout << x << " " << y << " --- "<< i << " " << j << std::endl;

                  if(x >= 0 && x < imgW && y >= 0 && y < imgH){
                     sumX += sobelX[k][l] * ((Mat) *img).at<Vec3b>(x, y)[0];
                     sumY += sobelY[k][l] * ((Mat) *img).at<Vec3b>(x, y)[0];
                  } 
               }
         }

         int v = (int) (sqrt(sumX * sumX + sumY * sumY) / 1448 * 256);  
         int a = max(v, treshold);

         if(a == treshold) 
            nx.at<Vec3b>(i, j) = Vec3b(0,0,0);
         else 
            nx.at<Vec3b>(i, j) = Vec3b(a,a,a);

      }
   }

   std::cout << mx << std::endl;

   *img = nx;

}

void bW(Mat *img){
   int rows = ((Mat) *img).rows;
   int cols = ((Mat) *img).cols;

   for(int i = 0; i < rows; i++){
      for(int j =0; j < cols; j++){
         Vec3b p = ((Mat) *img).at<Vec3b>(i, j);
         unsigned char gray = p[0] * 0.3 + p[1] * 0.58 + p[2] * 0.11;
         ((Mat) *img).at<Vec3b>(i, j) = Vec3b(gray,gray,gray);
      }
   }
}

void sobelSq(std::string iName, int th, bool blr, int bS, int bSg){
   
   name = iName;
   treshold = th;
   blur = blr;
   blurSize = bS;
   blurSigma = bSg;

   std::string image_path = samples::findFile("images/" + name);
   Mat img = imread(image_path, IMREAD_COLOR);


   bW(&img);
   if(blur) gaussBlur(&img);
   sobel(&img);

   imwrite("out/" + name, img);
}