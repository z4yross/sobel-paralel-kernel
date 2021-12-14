# FILE = main.cpp

# CC = g++

# LINKER_FLAGS =  -ldl  -lopencv_calib3d -lopencv_core -lopencv_imgcodecs
# NAME = main.o
# LBR = -I/usr/include/opencv2

# all: $(FILE) 
# 	$(CC) -c $(LBR) $(FILE) $(LINKER_FLAGS) -o $(NAME) 


CC = nvcc
# CFLAGS = 
SRCS = main.cu
PROG = main

OPENCV = `pkg-config opencv4 --cflags --libs`
LIBS = $(OPENCV)

all:
	rm -f $(PROG)
	$(CC) -o $(PROG) $(SRCS) $(LIBS)

# $(PROG):$(SRCS)
# 	$(CC) -o $(PROG) $(SRCS) $(LIBS)

