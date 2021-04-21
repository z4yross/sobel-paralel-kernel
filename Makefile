# FILE = main.cpp

# CC = g++

# LINKER_FLAGS =  -ldl  -lopencv_calib3d -lopencv_core -lopencv_imgcodecs
# NAME = main.o
# LBR = -I/usr/include/opencv2

# all: $(FILE) 
# 	$(CC) -c $(LBR) $(FILE) $(LINKER_FLAGS) -o $(NAME) 

CC = g++
CFLAGS = -g -Wall -ldl -lopencv_calib3d -lopencv_core -lopencv_imgcodecs 
SRCS = main.cpp
PROG = main

OPENCV = `pkg-config opencv4 --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)