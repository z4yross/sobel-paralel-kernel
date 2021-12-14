// make
// ./main -i t_640p.jpg -T 7 -B 10 -b 1 -k 7 -t 2 -m 1 -u 10

#include "sobelCUDA.cu"
#include <unistd.h>


void usage(const char *argv0){
   fprintf(stderr, "Usage: %s [-i nombreImagen][-T hilos][-B bloques][-t sobelTreshold][-b blur][-k blurSize][-s blurSigma][-m calcularPromedio][-u intentos][-h help-]\n", argv0);
   exit(EXIT_FAILURE);
}

int main(int argc, char **argv){
   std::string name = "gears.jpg";
   int treshold = 0;

   bool blur = true;
   int blurSize = 3;
   int blurSigma = 100;
   int blockSize = 1;
   bool dbg = false;
   int meanTr = 10;
   int blocks = 1;

   int opt;

   while ((opt = getopt(argc, argv, "i:T:B:t:b:k:s:m:u:h")) != -1){
      switch (opt){
         case 'i':
            name = optarg;
            break;
         case 'T':
            blockSize = atoi(optarg);
            break;
         case 'B':
            blocks = atoi(optarg);
            break;
         case 't':
            treshold = atoi(optarg);
            break;
         case 'b':
            blur = atoi(optarg) == 0 ? false : true;
            break;
         case 'k':
            blurSize = atoi(optarg);
            break;
         case 's':
            blurSigma = atoi(optarg);
            break;
         case 'm':
            dbg = atoi(optarg) >= 0 ? true : false;
            break;
         case 'u':
            meanTr = atoi(optarg);
            break;
         case 'h':
            usage(argv[0]);
         default:
            usage(argv[0]);
      }   
   }

   if(dbg){   
      for(int i = 0; i <= blockSize; i++){
         int p = pow(2, i);

         double sum = 0.0;

         for(int j = 0; j < meanTr; j++){
            sum += sobelSq(name, treshold, blur, blurSize, blurSigma, p, blocks);
         }

         sum /= meanTr;

         printf("%d threads %d blocks -> %fs mean\n", p, blocks, sum);
      }
   }else{
      float res = sobelSq(name, treshold, blur, blurSize, blurSigma, blockSize, blocks);
      printf("%d threads %d blocks -> %fs\n", blockSize, blocks, res);
   }


   return 0;
}
