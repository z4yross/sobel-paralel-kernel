# sobel-paralel-kernel

## Compilar

`make`

## Ejecutar
 `./main [-i nombreImagen][-T hilos][-t sobelTreshold][-b blur][-k blurSize][-s blurSigma][-m calcularPromedio][-u intentos][-h help]`

 - [-i nombreImagen]: nombre de la imagen dentro de la carpeta "images".
 - [-T hilos]: Si -m no se especifica, -T es la cantidad de hilos. De lo contrario, si -m > 0 -T es la cantidad de hilos maxima $(2^n)$.
 - [-t sobelTreshold]: Deja pasar unicamente los valores > -t al ejecutar sobel.
 - [-b blur]: Si es 0 no se realizara blur.
 - [-k blurSize]: tamano de kernel de blur.
 - [-s blurSigma]: Sigma utilizada para calcular el kernel del blur.
 - [-m calcularPromedio]: si > 0 Calcula el tiempo promedio con -u intentos con un maximo de $2^{-T}$ hilos.
 - [-u intentos]: cantidad de intentos para el promedio
 - [-h help]: Despliega la lista de flags.

## Integrantes
 - Andres Camilo Correa Romero - anccorrearo@unal.edu.co
 - Heidy Alayon Sastoque - halayons@unal.edu.co