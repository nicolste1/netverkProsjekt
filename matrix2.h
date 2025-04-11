#pragma once
#include <iostream>
#include <vector>

class Matrix2{
    private:
        int rows;
        int colums;
        std::vector<double> matrix; 
    public:
        Matrix2(int nrows, int ncolums);
        Matrix2();
        Matrix2(const Matrix2& rhs);
        Matrix2(const std::vector<std::vector<double>>& doubleVec);
        Matrix2(const std::vector<double>& Vec);
        void applyActivationFunc(std::string name);
        void sigmoidPrime();
        void setRandomValues();
        void setXavierValues(int dimIn, int dimOut);
        void setValue(double value);
        void roundedOutput();
        int argMax() const;
        Matrix2 transpose() const;
        const int getRows(){return rows; }
        const int getColums(){return colums; }
        friend std::ostream& operator<<(std::ostream& os, const Matrix2& rhs);
        friend std::istream& operator >>(std::istream& is,Matrix2& rhs);
        Matrix2 operator *(const Matrix2& rhs) const;
        Matrix2& operator *(double d);
        bool operator ==(const Matrix2& rhs);
        Matrix2 operator +(const Matrix2& rhs) const;
        Matrix2 operator -(const Matrix2& rhs) const;
        Matrix2& hademart(const Matrix2& rhs);
        Matrix2& operator =(Matrix2 rhs);
        double& operator () (size_t row, size_t colum); //Brukes ved endring 
        const double& operator () (size_t row, size_t colum) const; // Brukes ved å få verdien
};
