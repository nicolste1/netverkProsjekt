#include <matrix.h>
#include <cassert>
#include <random>

// Konstruktør og destruktør_____________________________________________

Matrix::Matrix(int nrows, int ncolums) : rows(nrows) , colums(ncolums){
    assert(rows > 0 && colums > 0);
    matrix = new double* [nrows];
    for(int i = 0; i < nrows; i++){
        matrix[i] = new double [colums];
    }
}

Matrix::Matrix() : rows(0), colums(0) {
    matrix = nullptr;
}

Matrix::Matrix(const std::vector<std::vector<double>>& doubleVec) : rows(doubleVec.size()), colums(doubleVec[0].size()){
    matrix = new double* [rows];
    for(int i = 0; i < rows; i++){
        matrix[i] = new double [colums];
        for(int j = 0; j <colums; j++){
            matrix[i][j] = doubleVec[i][j];
        }
    }
}

Matrix::Matrix(const std::vector<double>& Vec) : rows(Vec.size()), colums(1){
    matrix = new double* [rows];
    for(size_t i = 0; i < rows; i++){
        matrix[i] = new double [1];
        matrix[i][0] = Vec[i];
    }
}

Matrix::~Matrix(){
    for(int i = 0; i < rows; i++){
        delete[] matrix[i];
        matrix[i] = nullptr;
    }
    delete[] matrix;
    matrix = nullptr;
}

//Kopikonstruktør 
Matrix::Matrix(const Matrix& rhs) : colums(rhs.colums), rows(rhs.rows){
    matrix = new double* [rows];
    for(int i = 0; i < rows; i++){
        matrix[i] = new double [colums];
        for(int j = 0; j <colums; j++){
            matrix[i][j] = rhs.matrix[i][j];
        }
    }

}
//Medlemsfunskjoner________________________________________________________________

Matrix Matrix::transpose() const{
    Matrix out(this->colums, this->rows);
    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->colums; j++){
            out.matrix[j][i] = this->matrix[i][j];
        }
    }
    return out;
}
void Matrix::setRandomValues(){
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dist(0.0, 1.0);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < colums; j++){
            matrix[i][j] = dist(gen);
        }
    }
}

void Matrix::applyActivationFunc(std::string name){
    if(name == "sigmoid"){
        if(colums != 1){
            throw std::invalid_argument("Activation function not valid on matrix m x n where n != 1");
        }
        for(int i = 0; i < rows; i++){
            matrix[i][0] = 1.0 / (1.0 + std::exp(-matrix[i][0]));
        }
    }
    else{
        throw std::invalid_argument("Invalid activation function, at this moment we only have sigmoid");
    }
}

void Matrix::sigmoidPrime(){
    if(colums != 1){
        throw std::invalid_argument("sigmoidPrime not valid on matrix m x n where n != 1");
    }
    this->applyActivationFunc("sigmoid");
    for(int i = 0; i < rows; i++){
        matrix[i][0] = matrix[i][0] * (1 - matrix[i][0]);
    }
}

void Matrix::roundedOutput(){
    double localBiggest = 0;
    for(int i = 0; i < rows; i++){
        if(this->matrix[i][0] > localBiggest){
            this->matrix[i][0] = 1;
            }
        else{
            this->matrix[i][0] = 0;
            }
    }
}

int Matrix::argMax(){
    double localBiggest = this->matrix[0][0];
    int out = 0;
    for(int i = 1; i < rows; i++){
        if(this->matrix[i][0] > localBiggest){
            out = i;
            }
    }
    return out;
}

//Operator overlastning________________________________________

Matrix& Matrix::operator =(Matrix rhs){
    std::swap(rows,rhs.rows);
    std::swap(colums,rhs.colums);
    std::swap(matrix,rhs.matrix);
    return *this;
}


std::ostream& operator<<(std::ostream& os, const Matrix& rhs){
    for(int i = 0; i < rhs.rows; i++){
        for(int j = 0; j < rhs.colums; j++){
            os << rhs.matrix[i][j] << " ";
        }
        os << std::endl;
    }
    return os;
}


Matrix Matrix::operator *(const Matrix& rhs){ //Matrise ganger vector 
    if (rhs.rows != this->colums) {
        throw std::invalid_argument("Matrisemultiplikasjon går ikke da kolonner samsvarer ikke med antall rader");
    }
    Matrix out(this->rows,rhs.colums); //Gir ut en vector
    out.setValue(0);
    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < rhs.colums; j++){
            for(int k = 0; k < this->colums; k++){
                out.matrix[i][j] += this->matrix[i][k] * rhs.matrix[k][j];
            }
        }
    }
    return out;
}
Matrix Matrix::operator *(double d){
    Matrix out(this->rows,this->colums);
    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->colums; j++){
            out.matrix[i][j] = this->matrix[i][j] * d;
        }
    }
    return out;
}

Matrix Matrix::hademart(const Matrix& rhs){
    if (rhs.colums != 1) {
        throw std::invalid_argument("Du sendte inn en matrise og ikke vektor, kan ikke plusse matrise");
    }
    if (this->colums != 1) {
        throw std::invalid_argument("Du sendte inn en matrise og ikke vektor, kan ikke plusse matrise");
    }
    if (rhs.rows != this->rows) {
        throw std::invalid_argument("Vektorstørrelsen samsvarer ikke med antall kolonner");
    }
    Matrix out(this->rows,1);
    for(int i = 0; i < this->rows; i++){
        out.matrix[i][0] = this->matrix[i][0] * rhs.matrix[i][0];
    }
    return out;
}

Matrix Matrix::operator+(const Matrix& rhs){
    if (rhs.rows != this->rows || rhs.colums != this->colums) {
        throw std::invalid_argument("Matrisene er ikke av samme størelse og kan derfor ikke adderes sammen");
    }
    Matrix out(this->rows, this->colums); //Gir ut av samme dim 

    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->colums; j++){
            out.matrix[i][j] = this->matrix[i][j] + rhs.matrix[i][j];
        }
    }
    return out;
}

Matrix Matrix::operator-(const Matrix& rhs){
    if (rhs.rows != this->rows || rhs.colums != this->colums) {
        throw std::invalid_argument("Matrisene er ikke av samme størelse og kan derfor ikke adderes sammen");
    }
    Matrix out(this->rows, this->colums); //Gir ut av samme dim 

    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->colums; j++){
            out.matrix[i][j] = this->matrix[i][j] - rhs.matrix[i][j];
        }
    }
    return out;
    
}

bool Matrix::operator ==(const Matrix& rhs){
    if(this->rows != rhs.rows){
        return false;
    }
    if(this->colums != rhs.colums){
        return false;
    }
    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->colums; j++){
            if(this->matrix[i][j] != rhs.matrix[i][j])
            return false;
        }
    }
    return true;
}
//Ikke essensielle funksjoner i bruk for test____________________________________
void Matrix::setValue(int value){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < colums; j++){
            matrix[i][j] = value;
        }
    }
}