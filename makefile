CC 	= g++
CFLAGS 	= -std=c++11 -Wall -fopenmp -c
RFLAGS 	= -std=c++11 -Wall -fopenmp -o
NN	= NeuralNetDriver.exe
KF	= NeuralNetArchitectureDriver.exe
MC	= MonteCarloDriver.exe

default: comp_all

#MAKE COMPILER OPTIONS

comp_all:	NeuralNetDriver.o MonteCarloDriver.o NeuralNetArchitectureDriver.o Network.o Node.o Validation.o
		$(CC) $(RFLAGS) $(NN) NeuralNetDriver.o Node.o Network.o
		$(CC) $(RFLAGS) $(KF) NeuralNetArchitectureDriver.o Validation.o Node.o Network.o
		$(CC) $(RFLAGS) $(MC) MonteCarloDriver.o Validation.o Node.o Network.o
comp_mc:	MonteCarloDriver.o Validation.o Network.o Node.o
		$(CC) $(RFLAGS) $(MC) MonteCarloDriver.o Validation.o Network.o Node.o
comp_kf:	NeuralNetArchitectureDriver.o Validation.o Network.o Node.o
		$(CC) $(RFLAGS) $(KF) NeuralNetArchitectureDriver.o Validation.o Network.o Node.o
comp_nn:	NeuralNetDriver.o Network.o Node.o
		$(CC) $(RFLAGS) $(NN) NeuralNetDriver.o Network.o Node.o

#MAKE COMMANDS
clean:
	$(RM) *.o *.exe
Network.o:		Network.cpp Network.h Node.h
			$(CC) $(CFLAGS) Network.cpp
Node.o:			Node.cpp Node.h
			$(CC) $(CFLAGS) Node.cpp
NeuralNetDriver.o:	NeuralNetDriver.cpp Network.h
			$(CC) $(CFLAGS) NeuralNetDriver.cpp
NeuralNetArchitectureDriver.o:		NeuralNetArchitectureDriver.cpp Network.h Node.h Validation.h
			$(CC) $(CFLAGS) NeuralNetArchitectureDriver.cpp
MonteCarloDriver.o:	MonteCarloDriver.cpp Network.h Node.h Validation.h
			$(CC) $(CFLAGS) MonteCarloDriver.cpp
Validation.o:		Validation.cpp Validation.h Network.h
			$(CC) $(CFLAGS) Validation.cpp
