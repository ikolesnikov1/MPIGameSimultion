#include "mpi.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
 
int iterations = 2000;
int X = 350;
int Y = 350;
 
void calcNextState(const int *currentState, int *nextState, int rowsToSkip, int rowsNumber, int N) {
    for (int i = rowsToSkip; i < rowsToSkip + rowsNumber; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int vertical = -1; vertical <= 1; vertical++) {
                for (int horiz = -1; horiz <= 1; horiz++) {
                    if (horiz != 0 || vertical != 0)
                        sum += currentState[(i + vertical) * N + (j + horiz + N) % N];
                }
            }
 
            if (currentState[i * N + j] && (sum < 2 || sum > 3))
                nextState[i * N + j] = 0;
            else if (!currentState[i * N + j] && sum == 3)
                nextState[i * N + j] = 1;
            else
                nextState[i * N + j] = currentState[i * N + j];
        }
    }
}
 
void calcBreakFlags(const int *currentState, std::vector<int*> &allStates, int x, int y, int *breakFlags) {
    int i = 0;
    for (int j = 0; j != allStates.size(); j++) {
        int flag = 0;
        for (int k = 0; k < x * y; k++) {
            if ((allStates[j] + y)[k] != currentState[k]) {
                flag = 1;
                breakFlags[i] = 0;
                break;
            }
        }
 
        if(!flag) breakFlags[i] = 1;
        ++i;
    }
}
 
int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    int* cellsArray = NULL;
    if (rank == 0) {
        cellsArray = (int*)calloc(X * Y, sizeof(int));
        cellsArray[0 * Y + 1] = 1;cellsArray[1 * Y + 2] = 1; cellsArray[2 * Y + 0] = 1; cellsArray[2 * Y + 1] = 1; cellsArray[2 * Y + 2] = 1;
    }
 
    // Массив с числом эл-ов на процесс
    int elementsForProc[size];
    int add = X % size;
    for (int i = 0; i < size; i++) {
        elementsForProc[i] = X / size * Y + (--add >= 0 ? Y : 0);
    }
 
    //смещение для scatterv
    int displs[size], countOfElements = 0;
    for (int i = 0; i < size; i++) {
        displs[i] = countOfElements;
        countOfElements += elementsForProc[i];
    }
 
    // новое значение - в nextState, сравнивать с curState
    int* curState = (int*)calloc(elementsForProc[rank] + 2 * Y, sizeof(int));
    int* nextState = (int*)calloc(elementsForProc[rank] + 2 * Y, sizeof(int));
    // Если полученное состояние = старому => 1 в вектор для процесса.
    // Потом собираем со всех и ставим 0 / 1 в вектор для всех - условие выхода
    int* breakFlagsForProc = (int*)calloc(iterations, sizeof(int));
    int* breakFlags = (int*)calloc(iterations, sizeof(int));
    // 1 эл-т вектора хранит матрицу предыдущих состояний
    std::vector<int*> allStates;
 
    double timeStart = MPI_Wtime();
 
    //пропустить 1 строку (+Y) чтобы писать в наш блок
    MPI_Scatterv(cellsArray, elementsForProc, displs, MPI_INT, curState + Y, elementsForProc[rank], MPI_INT, 0, MPI_COMM_WORLD);
 
    // ID события для проверки окончания
    MPI_Request request1, request2, request3, request4, request5;
    int numOfIterations = 0;
    for(int i = 0; i < iterations; i++) {
        numOfIterations++;
 
        //1, 2
        MPI_Isend(curState + Y, Y, MPI_INT, (rank - 1 + size) % size, 0, MPI_COMM_WORLD, &request1);
        MPI_Isend(curState + elementsForProc[rank], Y, MPI_INT, (rank + 1 + size) % size, 1, MPI_COMM_WORLD, &request2);
 
        //3, 4
        MPI_Irecv(curState, Y, MPI_INT, (rank - 1 + size) % size, 1, MPI_COMM_WORLD, &request3);
        MPI_Irecv(curState + elementsForProc[rank] + Y, Y, MPI_INT, (rank + 1 + size) % size, 0, MPI_COMM_WORLD, &request4);
 
        //5
        // Сравниваем только наши состояния
        calcBreakFlags(curState + Y, allStates, elementsForProc[rank] / Y, Y, breakFlagsForProc);
 
        //6, 7 - сборка breakFlags + пропускаем строки соседние с чужими (берем независимые)
        MPI_Iallreduce(breakFlagsForProc, breakFlags, i, MPI_INT, MPI_LAND, MPI_COMM_WORLD, &request5);
        calcNextState(curState + Y, nextState + Y, 1, elementsForProc[rank] / Y - 2, Y);
 
        //8, 9, 10
        MPI_Wait(&request1, MPI_STATUS_IGNORE);
        MPI_Wait(&request3, MPI_STATUS_IGNORE);
        // Проверка 1ой строки реального блока, тк зависит от чужой
        calcNextState(curState + Y, nextState + Y, 0, 1, Y);
 
        //11, 12
        MPI_Wait(&request2, MPI_STATUS_IGNORE);
        MPI_Wait(&request4, MPI_STATUS_IGNORE);
 
        //13, 14
        calcNextState(curState + Y, nextState + Y, elementsForProc[rank] / Y - 1, 1, Y);
        MPI_Wait(&request5, MPI_STATUS_IGNORE);
 
        //15
        for (int j = 0; j < i; j++) {
            if (breakFlags[j])
                break;
        }
 
        allStates.push_back(curState);
        curState = nextState;
        nextState = (int*)calloc(elementsForProc[rank] + 2 * Y, sizeof(int));
    }
 
    double timeEnd = MPI_Wtime();
 
    if (rank == 0) {
        std::cout << "iterations: " << numOfIterations << std::endl;
        std::cout << "total time: " << timeEnd - timeStart << std::endl;
    }
 
    free(cellsArray); free(curState); free(nextState); free(breakFlagsForProc); free(breakFlags);
    MPI_Finalize();
    return 0;
}
