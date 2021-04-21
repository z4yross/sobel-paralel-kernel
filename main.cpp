#include "sobel.cpp"

int main(){
   std::string name = "gears.jpg";
   int treshold = 0;

   bool blur = true;
   int blurSize = 3;
   int blurSigma = 100;

   sobelSq(name, treshold, blur, blurSize, blurSigma);

   return 0;
}