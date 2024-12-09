#include "pch.h"
#include "DWT.h"
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <limits>
#include <iostream>
#include <ctime>

std::vector<double> initMatrix(size_t size1, size_t size2) {
	size_t matrixSize = (size1 + 1) * (size2 + 1);
	std::vector<double> matrix(matrixSize, std::numeric_limits<double>::infinity());
	matrix[0] = 0.0;
	return matrix;
}

double getVal(std::vector<double>& matrix, size_t size2, size_t i, size_t j) {
	size_t index = i * size2 + j;
	return matrix[index];
}

void setVal(std::vector<double>& matrix, double val, size_t size2, size_t i, size_t j) {
	size_t index = i * size2 + j;
	matrix[index] = val;
	return;
}

double distance(std::vector<double> coord1, std::vector<double> coord2) {
	return sqrt(pow(coord1[0] - coord2[0], 2) + pow(coord1[1] - coord2[1], 2));
}

double find_cost(const double* series1, size_t series1Length, const double* series2, size_t series2Length) {
	std::vector<std::vector<double>> s1(series1Length, std::vector<double>(2));
	std::vector<std::vector<double>> s2(series2Length, std::vector<double>(2));

	for (size_t i = 0; i < series1Length; i++) {
		s1[i][0] = series1[i * 2];
		s1[i][1] = series1[i * 2 + 1];
	}

	for (size_t i = 0; i < series2Length; i++) {
		s2[i][0] = series2[i * 2];
		s2[i][1] = series2[i * 2 + 1];
	}

	std::vector<double> costMatrix = initMatrix(series1Length, series2Length);

	for (unsigned int i = 1; i <= series1Length; i++) {
		for (unsigned int j = 1; j <= series2Length; j++) {
			double costVal = distance(s1[i - 1], s2[j - 1]);
			double insertion = getVal(costMatrix, series2Length, i - 1, j);
			double deletion = getVal(costMatrix, series2Length, i, j - 1);
			double match = getVal(costMatrix, series2Length, i - 1, j - 1);
			double minVal = insertion;
			minVal = deletion < minVal ? deletion : minVal;
			minVal = match < minVal ? match : minVal;
			setVal(costMatrix, costVal + minVal, series2Length, i, j);
		}
	}
	return getVal(costMatrix, series2Length, series1Length, series2Length);
}
