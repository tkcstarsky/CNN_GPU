CC = g++
CCFLAGS = -std=c++11

main: main.o CNN.o
	$(CC) -o main main.o CNN.o $(CCFLAGS)

main.o: CNN.h

CNN.o: CNN.h

.PHONY: clean

clean:
	rm main *.o
