#pragma once
#include <iostream>
#include <vector>

class Matrix{
    private:
        int rows;
        int colums;
        double** matrix;
    public:
        Matrix(int nrows, int ncolums);
        Matrix();
        Matrix(const Matrix& rhs);
        ~Matrix();
        Matrix(const std::vector<std::vector<double>>& doubleVec);
        Matrix(const std::vector<double>& Vec);
        void applyActivationFunc(std::string name);
        void sigmoidPrime();
        void setRandomValues();
        void setValue(int value);
        void roundedOutput();
        int argMax();
        Matrix transpose() const;
        const int getRows(){return rows; }
        const int getColums(){return colums; }
        friend std::ostream& operator<<(std::ostream& os, const Matrix& rhs);
        Matrix operator *(const Matrix& rhs);
        Matrix operator *(double d);
        bool operator ==(const Matrix& rhs);
        Matrix operator +(const Matrix& rhs);
        Matrix operator -(const Matrix& rhs);
        Matrix hademart(const Matrix& rhs);
        Matrix& operator =(Matrix rhs);
};
