CC = g++
CCFLAGS = 

main: main.o CNN.o
	$(CC) -o main main.o CNN.o $(CCFLAGS)

main.o: CNN.h

CNN.o: CNN.h

.PHONY: clean

clean:
	rm main *.o
