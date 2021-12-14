#include <iostream>

#include <mpi.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <unistd.h>

using namespace cv;

void usage(const char *argv0)
{
    fprintf(stderr, "Usage: %s [-i nombreImagen][-T hilos][-B bloques][-t sobelTreshold][-b blur][-k blurSize][-s blurSigma][-m calcularPromedio][-u intentos][-h help-]\n", argv0);
    exit(EXIT_FAILURE);
}

void gaussKernel(int size, double sigma, int K, double *out)
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

int main(int argc, char **argv)
{

    Mat im;
    Mat image;
    size_t imageTotalSize;
    int channels;

    size_t imagePartialSize;
    uchar *partialBuffer;
    uchar *data;

    Mat outImage;

    int img_cols;
    int img_rows;

    int kS = 3;

    int treshold = 10;

    int blurSize = 3;
    int blurSigma = 3;

    const double sobelX[3][3] = {
        {1.0, 0.0, -1.0},
        {2.0, 0.0, -2.0},
        {1.0, 0.0, -1.0}};

    const double sobelY[3][3] = {
        {1.0, 2.0, 1.0},
        {0.0, 0.0, 0.0},
        {-1.0, -2.0, -1.0}};

    double kernel[blurSize];

    std::string name = "gears.jpg";

    int opt;

    while ((opt = getopt(argc, argv, "i:t:k:s:h")) != -1)
    {
        switch (opt)
        {
        case 'i':
            name = optarg;
            break;
        case 't':
            treshold = atoi(optarg);
            break;
        case 'k':
            blurSize = atoi(optarg);
            break;
        case 's':
            blurSigma = atoi(optarg);
            break;
        case 'h':
            usage(argv[0]);
        default:
            usage(argv[0]);
        }
    }

    // -----------------------------------------------------------------------------------------------------------------------------------------------

    MPI_Init(&argc, &argv);

    int size;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {   
        std::cout << std::endl;

        im = imread("images/" + name, IMREAD_UNCHANGED);

        if (im.empty())
        {
            std::cerr << "No existe la imagen en la ruta: images/" << name << std::endl;
            return -1;
        }

        cvtColor(im, image, COLOR_RGB2GRAY);

        img_cols = image.cols;
        img_rows = image.rows;

        gaussKernel(blurSize, blurSigma, 1, kernel);

        channels = image.channels();
        data = image.data;

        imageTotalSize = image.cols * image.rows;

        if (image.total() % size)
        {
            std::cerr << "No se puede dividir la imagen en un numero par" << std::endl;
            return -2;
        }

        imagePartialSize = imageTotalSize / size;

        std::cout << "La imagen sera dividida en bloques de " << imagePartialSize << " bytes" << std::endl;
    }

    // -----------------------------------------------------------------------------------------------------------------------------------------------

    MPI_Bcast(&imageTotalSize, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imagePartialSize, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&img_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank != 0)
    {
        data = new uchar[imageTotalSize];
    }

    partialBuffer = new uchar[imagePartialSize];

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(data, imageTotalSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    MPI_Scatter(image.data, imagePartialSize, MPI_UNSIGNED_CHAR, partialBuffer, imagePartialSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // -----------------------------------------------------------------------------------------------------------------------------------------------

    for (size_t idx = 0; idx < imagePartialSize; idx += channels)
    {
        int i = (idx + (imagePartialSize * rank)) % img_cols;
        int j = (idx + (imagePartialSize * rank)) / img_cols;

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
                    sum += data[y * img_cols + x] * kV;
                }
            }
        }

        partialBuffer[idx] = sum;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t idx = 0; idx < imagePartialSize; idx += channels)
    {
        int i = (idx + (imagePartialSize * rank)) % img_cols;
        int j = (idx + (imagePartialSize * rank)) / img_cols;

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
                    sumX += sobelX[k][l] * data[y * img_cols + x];
                    sumY += sobelY[k][l] * data[y * img_cols + x];
                }
            }
        }

        int v = (int)(sqrt(sumX * sumX + sumY * sumY) / 1448 * 256);
        int a = v > treshold ? v : treshold;

        // partialBuffer[idx] = v;

        if (a == treshold)
            partialBuffer[idx] = 0;
        else
            partialBuffer[idx] = 255;
    }
    // -----------------------------------------------------------------------------------------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        outImage = Mat(image.size(), image.type());
    }

    MPI_Gather(partialBuffer, imagePartialSize, MPI_UNSIGNED_CHAR, outImage.data, imagePartialSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        imwrite("out/" + name, outImage);
    }

    delete[] partialBuffer;

    MPI_Finalize();
}
