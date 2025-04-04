#include <matrix2.h>
#include <cassert>
#include <random>

// Konstruktør og destruktør_____________________________________________

Matrix2::Matrix2(int nrows, int ncolums) : rows(nrows) , colums(ncolums), matrix(nrows *ncolums, 0.0){
    assert(rows > 0 && colums > 0);

}

Matrix2::Matrix2() : rows(0), colums(0) {
}

Matrix2::Matrix2(const std::vector<std::vector<double>>& doubleVec) : rows(doubleVec.size()), colums(doubleVec[0].size()){
    for(int row = 0; row < rows; row++){
        for(int colum = 0; colum < colums; colum++){
            this->matrix.push_back((doubleVec.at(row)).at(colum));
        }
    }
}

Matrix2::Matrix2(const std::vector<double>& Vec) : rows(Vec.size()), colums(1){
    matrix = Vec;
}

//Kopikonstruktør 
Matrix2::Matrix2(const Matrix2& rhs) : colums(rhs.colums), rows(rhs.rows){
    matrix = rhs.matrix;
}
//Medlemsfunskjoner________________________________________________________________

Matrix2 Matrix2::transpose() const{
    Matrix2 out(this->colums, this->rows);
    const Matrix2& local = *this;
    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->colums; j++){
            out(j,i) = local(i,j);
        }
    }
    return out;
}
void Matrix2::setRandomValues(){
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dist(0.0, 1.0);
    for(int i = 0; i < rows * colums; i++){
        double random = dist(gen);
        matrix.at(i) = random;
    }
}

void Matrix2::setXavierValues(int dimIn, int dimOut){
    std::random_device rd;
    std::mt19937 gen(rd()); 
    double Xavierlim = std::sqrt(6.0 / (dimIn + dimOut));
    std::uniform_real_distribution<> dist(-Xavierlim, Xavierlim);
    for(int i = 0; i < rows * colums; i++){
        double random = dist(gen);
        matrix.at(i) = random;
    }
}



void Matrix2::applyActivationFunc(std::string name){
    if(name == "sigmoid"){
        if(colums != 1){
            throw std::invalid_argument("Activation function not valid on matrix m x n where n != 1");
        }

        for(int i = 0; i < rows; i++){
            matrix.at(i) = 1.0 / (1.0 + std::exp(-matrix.at(i)));
        }
    }
    else{
        throw std::invalid_argument("Invalid activation function, at this moment we only have sigmoid");
    }
}

void Matrix2::sigmoidPrime(){
    if(colums != 1){
        throw std::invalid_argument("sigmoidPrime not valid on matrix m x n where n != 1");
    }
    this->applyActivationFunc("sigmoid");
    for(int i = 0; i < rows; i++){
        matrix.at(i) = matrix.at(i) * (1 - matrix.at(i));
    }
}

void Matrix2::roundedOutput(){
    if(colums != 1){
        throw std::invalid_argument("roundedOutput not valid on matrix m x n where n != 1, only on vector");
    }
    size_t idxBiggest = 0;
    double localBiggest = this->matrix.at(0);
    for(int i = 1; i < rows; i++){
        if(this->matrix.at(i) > localBiggest){
            localBiggest = this->matrix.at(i);
            idxBiggest = i;
        }
        this->matrix.at(i) = 0;
    }
    this->matrix.at(idxBiggest) = 1;
}

int Matrix2::argMax() const{
    if(colums != 1){
        throw std::invalid_argument("argMax not valid on matrix m x n where n != 1, only on vector");
    }
    double localBiggest = this->matrix.at(0);
    int out = 0;
    for(int i = 1; i < rows; i++){
        if(this->matrix.at(i) > localBiggest){
            localBiggest = this->matrix.at(i);
            out = i;
        }
    }
    return out;
}

//Operator overlastning________________________________________

double& Matrix2::operator () (size_t row, size_t colum){
    return this->matrix.at((this->colums) * row + colum);
}

const double& Matrix2::operator () (size_t row, size_t colum) const{
    return this->matrix.at((this->colums) * row + colum);
}


Matrix2& Matrix2::operator =(Matrix2 rhs){
    std::swap(rows,rhs.rows);
    std::swap(colums,rhs.colums);
    std::swap(matrix,rhs.matrix);
    return *this;
}


std::ostream& operator<<(std::ostream& os, const Matrix2& rhs){
    for(int row = 0; row < rhs.rows; row++){
        for(int colum = 0; colum < rhs.colums; colum++){
            os << rhs.matrix.at(row * rhs.colums + colum) << " ";
        }
        os << std::endl;
    }
    return os;
}


Matrix2 Matrix2::operator *(const Matrix2& rhs) const{ //Matrise ganger vector 
    if (rhs.rows != this->colums) {
        throw std::invalid_argument("Matrisemultiplikasjon går ikke da kolonner samsvarer ikke med antall rader");
    }
    const Matrix2& local = *this;
    Matrix2 out(this->rows,rhs.colums); //Gir ut en vector
    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < rhs.colums; j++){
            for(int k = 0; k < this->colums; k++){
                out(i,j) += local(i,k) * rhs(k,j);
            }
        }
    }
    return out;
}

Matrix2& Matrix2::operator *(double d){
    for(int i = 0; i < rows * colums; i++){
        this->matrix.at(i);
    }
    return *this;
}

Matrix2& Matrix2::hademart(const Matrix2& rhs){
    if (rhs.colums != 1) {
        throw std::invalid_argument("Du sendte inn en matrise og ikke vektor, kan ikke plusse matrise");
    }
    if (this->colums != 1) {
        throw std::invalid_argument("Du sendte inn en matrise og ikke vektor, kan ikke plusse matrise");
    }
    if (rhs.rows != this->rows) {
        throw std::invalid_argument("Vektorstørrelsen samsvarer ikke med antall kolonner");
    }
    for(int i = 0; i < this->rows; i++){
        this->matrix.at(i) = this->matrix.at(i) * rhs.matrix.at(i);
    }
    return *this;
}

//Utifra network kode kunne man implementert + til å returnere referanse og ikke lake kopi, 
//men dette bryter med standard så ikke implementert, og unødvendig da RVO fikser 
Matrix2 Matrix2::operator + (const Matrix2& rhs) const{
    if (rhs.rows != this->rows || rhs.colums != this->colums) {
        throw std::invalid_argument("Matrisene er ikke av samme størelse og kan derfor ikke adderes sammen");
    }

    Matrix2 out(this->rows, this->colums); //Gir ut av samme dim 
    const Matrix2& local = *this;

    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->colums; j++){
            out(i,j) = local(i,j) + rhs(i,j);
        }
    }

    return out;
}

Matrix2 Matrix2::operator-(const Matrix2& rhs) const{
    if (rhs.rows != this->rows || rhs.colums != this->colums) {
        throw std::invalid_argument("Matrisene er ikke av samme størelse og kan derfor ikke adderes sammen");
    }

    const Matrix2& local = *this;
    Matrix2 out(this->rows, this->colums); //Gir ut av samme dim 

    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->colums; j++){
            out(i,j) = local(i,j) - rhs(i,j);
        }
    }

    return out;
}

bool Matrix2::operator ==(const Matrix2& rhs){
    if(this->rows != rhs.rows){
        return false;
    }
    if(this->colums != rhs.colums){
        return false;
    }
    for(int i = 0; i < rows * colums; i++){
        if(this->matrix.at(i) != rhs.matrix.at(i)){
            return false;
        }
    }
    return true;
}
//Ikke essensielle funksjoner i bruk for test____________________________________
void Matrix2::setValue(double value){
    for(int i = 0; i < rows * colums; i++){
        matrix.at(i) = value;
    }
}




