#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <iostream>

#include <sys/time.h>

#include <pthread.h>

using namespace cv;

struct argsSobel {
   Mat *img;
   Mat *out;
   int idxI;
   int idxF;
   int th;
};

struct argsBlur {
   Mat *img;
   Mat *out;
   int idxI;
   int idxF;
   int blurSize;
   double** kernel;
};

std::string name;

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

void *gaussBlur(void *arg){
   argsBlur prms = *(argsBlur *) arg;
   Mat *img = prms.img;
   Mat *out = prms.out;
   int idxI = prms.idxI;
   int idxF = prms.idxF;
   double** kernel = prms.kernel;
   int kS = prms.blurSize;

   // printf("%d %d %d %d %d %d -- AA\n", out, out, out, out, out, out);
   // double** kernel = gaussKernel(3, 100, 1);
   // int kS = 3;

   int imgW = ((Mat) *img).cols;
   int imgH = ((Mat) *img).rows;

   Mat nx = ((Mat) *img).clone();
   Mat mask = Mat(imgH, imgW, nx.type(), 0.0);

   for (int idx = idxI; idx < idxF; idx++){
      int i = idx % imgW;
      int j = idx / imgW;

      float sum = 0.0;

      for(int k = 0; k < kS; k++){
            for(int l = 0; l < kS; l++){
               int x = i + (k - kS / 2);
               int y = j + (l - kS / 2);          

               if(x >= 0 && x < imgW && y >= 0 && y < imgH){
                  double kV = kernel[k][l];
                  sum += ((Mat) *img).at<Vec3b>(y, x)[0] * kV;

               } 
            }
      }
      
      nx.at<Vec3b>(j, i) = Vec3b(sum,sum,sum);
      mask.at<Vec3b>(j, i) = Vec3b(1, 1, 1);

   }
   nx.copyTo(((Mat)*out), mask);
   return NULL;
}


void *sobel(void *arg){
   argsSobel prms = *(argsSobel *) arg;
   Mat *img = prms.img;
   Mat *out = prms.out;
   int idxI = prms.idxI;
   int idxF = prms.idxF;
   int treshold = prms.th;

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

   int imgW = ((Mat) *img).cols;
   int imgH = ((Mat) *img).rows;

   Mat nx = ((Mat) *img).clone();
   Mat mask = Mat(imgH, imgW, nx.type(), 0.0);

   for (int idx = idxI; idx < idxF; idx++){

      int i = idx % imgW;
      int j = idx / imgW;

      double sumX = 0.0;
      double sumY = 0.0;

      for(int k = 0; k < kS; k++){
         for(int l = 0; l < kS; l++){
            int x = i + (k - kS / 2);
            int y = j + (l - kS / 2);          
            

            if(x >= 0 && x < imgW && y >= 0 && y < imgH){
               sumX += sobelX[k][l] * ((Mat) *img).at<Vec3b>(y, x)[0];
               sumY += sobelY[k][l] * ((Mat) *img).at<Vec3b>(y, x)[0];
            } 
         }
      }

      int v = (int) (sqrt(sumX * sumX + sumY * sumY) / 1448 * 256);  
      int a = max(v, treshold);

      if(a == treshold) 
         nx.at<Vec3b>(j, i) = Vec3b(0,0,0);
      else 
         nx.at<Vec3b>(j, i) = Vec3b(255,255,255);

      mask.at<Vec3b>(j, i) = Vec3b(1, 1, 1);

   }
   nx.copyTo(((Mat)*out), mask);
   return NULL;
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

float sobelSq(std::string iName, int th, bool blr, int bS, int bSg, int p){
   name = iName;

   struct timeval tval_before, tval_after, tval_result;
   gettimeofday(&tval_before, NULL);

   std::string image_path = samples::findFile("images/" + name);
   Mat img = imread(image_path, IMREAD_COLOR);

   Mat out = img.clone();

   bW(&img);

   // ----------------- BLUR-----------------
   if (blr){      
      double** kernel = gaussKernel(bS, bSg, 1);
      int rDelta = (img.cols * img.rows - 1) / p;
      int r = 0;

      pthread_t threadsB[p];
      argsBlur * prmsB;
      prmsB = (argsBlur *) malloc(p * 1024 * sizeof(argsBlur));

      Mat imgs[p];

      for(int i = 0; i < p; i++){
         int rI = r;
         int rF = min(r + rDelta, img.cols * img.rows - 1) ;

         // printf("%d - %d\n", r + rDelta, img.cols * img.rows -1);

         imgs[i] = img.clone();

         (prmsB + i * 1024) -> idxI = rI;
         (prmsB + i * 1024) -> idxF = rF;
         (prmsB + i * 1024) -> img = &imgs[i];
         (prmsB + i * 1024) -> out = &out;
         (prmsB + i * 1024) -> blurSize = bS;
         (prmsB + i * 1024) -> kernel = kernel;

         int res = pthread_create(&threadsB[i], NULL, gaussBlur, (prmsB + i * 1024));

         if(res != 0) exit(-1);

         r += rDelta;

         // pthread_join(threadsB[i], NULL);
      }

      for(int t = 0; t < p; t++)  pthread_join(threadsB[t], NULL);

      free(prmsB);
   }


   img = out.clone();
   

   // ----------------- SOBEL-----------------

   int rDelta = (img.cols * img.rows - 1) / p;
   int r = 0;

   pthread_t threads[p];

   argsSobel * prms;
   prms = (argsSobel *) malloc(p * 1024 * sizeof(argsSobel));

   Mat imgs[p];

   for(int i = 0; i < p; i++){
      int rI = r;
      int rF = min(r + rDelta, img.cols * img.rows -1);

      imgs[i] = img.clone();
      
      (prms + i * 1024) -> idxI = rI;
      (prms + i * 1024) -> idxF = rF;
      (prms + i * 1024) -> img = &imgs[i];
      (prms + i * 1024) -> out = &out;
      (prms + i * 1024) -> th = th;

      int res = pthread_create(&threads[i], NULL, sobel, (prms + i * 1024));

      if(res != 0) exit(-1);

      r += rDelta;
   }

   for(int t = 0; t < p; t++)  pthread_join(threads[t], NULL);
      

   gettimeofday(&tval_after, NULL);
   timersub(&tval_after, &tval_before, &tval_result);
   // printf("%d threads -> %ld.%06lds\n", p, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

   imwrite("out/" + name, out);

   free(prms);

   // String res = tval_result.tv_sec + "." + tval_result.tv_usec;
   // std::cout << res << std::endl;
   return (long int)tval_result.tv_sec + (long int)tval_result.tv_usec * 0.000001;
}