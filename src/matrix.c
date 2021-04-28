#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if(rows<1 || cols<1){
      return -1;
    }
    double *data = calloc(rows*cols, sizeof(double));
    matrix *m = malloc(sizeof(matrix));
    if(!m || !data){
      return -1;
    }
    m->rows = rows;
    m->cols = cols;
    m->data = data;
    m->ref_cnt = 1;
    m->parent = NULL;
    *mat = m;
    return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both are non-positive or if any
 * call to allocate memory in this function fails.
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if(rows<1 || cols<1){
      return -1;
    }
    matrix *m = malloc(sizeof(matrix));
    if(!m) {
      return -1;
    }
    m->rows = rows;
    m->cols = cols;
    m->data = from->data + offset;
    from->ref_cnt += 1;
    m->parent = from;
    *mat = m;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if(!mat){
      return;
    }
    if(mat->parent == NULL && mat->ref_cnt == 1){
      free(mat->data);
    }
    if(mat->parent != NULL){
	mat->parent->ref_cnt -= 1;
	if(mat->parent->ref_cnt == 0){
	  deallocate_matrix(mat->parent);
	}
    }
    if(mat->ref_cnt == 1){
	free(mat);
    } else {
      mat->ref_cnt -= 1;
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    /* TODO: YOUR CODE HERE */
    double res = (mat->data)[row*(mat->cols) + col];
    return res;
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
    (mat->data)[row*(mat->cols) + col] = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */
    int size = (mat->rows)*(mat->cols);
    for(int i = 0; i < size/4 * 4; i+=4){
      (mat->data)[i] = val;
      (mat->data)[i+1] = val;
      (mat->data)[i+2] = val;
      (mat->data)[i+3] = val;
    }
    for(int i = size/4 * 4; i < size; i++){
      (mat->data)[i] = val;
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    int size = (result->rows)*(result->cols);
    double *rd = result->data;
    double *d1 = mat1->data;
    double *d2 = mat2->data;
    __m256d a;
    __m256d b;
#pragma omp parallel for private(a,b)
    for(int i = 0; i < size/16 * 16; i+=16){
      a = _mm256_loadu_pd(d1+i);
      b = _mm256_loadu_pd(d2+i);
      _mm256_storeu_pd(rd + i, _mm256_add_pd(a, b));
      a = _mm256_loadu_pd(d1+i+4);
      b = _mm256_loadu_pd(d2+i+4);
      _mm256_storeu_pd(rd + i+4, _mm256_add_pd(a, b));
      a = _mm256_loadu_pd(d1+i+8);
      b = _mm256_loadu_pd(d2+i+8);
      _mm256_storeu_pd(rd + i+8, _mm256_add_pd(a, b));
      a = _mm256_loadu_pd(d1+i+12);
      b = _mm256_loadu_pd(d2+i+12);
      _mm256_storeu_pd(rd + i+12, _mm256_add_pd(a, b));
    }
    for(int i = size/16 * 16; i < size; i++){
	rd[i] = d1[i] + d2[i];
    }
    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    int size = (result->rows)*(result->cols);
    double *rd = result->data;
    double *d1 = mat1->data;
    double *d2 = mat2->data;
    __m256d a;
    __m256d b;
#pragma omp parallel for private(a,b)
    for(int i = 0; i < size/16 * 16; i+=16){
      a = _mm256_loadu_pd(d1+i);
      b = _mm256_loadu_pd(d2+i);
      _mm256_storeu_pd(rd + i, _mm256_sub_pd(a, b));
      a = _mm256_loadu_pd(d1+i+4);
      b = _mm256_loadu_pd(d2+i+4);
      _mm256_storeu_pd(rd + i+4, _mm256_sub_pd(a, b));
      a = _mm256_loadu_pd(d1+i+8);
      b = _mm256_loadu_pd(d2+i+8);
      _mm256_storeu_pd(rd + i+8, _mm256_sub_pd(a, b));
      a = _mm256_loadu_pd(d1+i+12);
      b = _mm256_loadu_pd(d2+i+12);
      _mm256_storeu_pd(rd + i+12, _mm256_sub_pd(a, b));
    }
    for(int i = size/16 * 16; i < size; i++){
        rd[i] = d1[i] - d2[i];
    }
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    double sum;
    matrix *mat2T;
    double *rd = result->data;
    double *d1 = mat1->data;
    int cols1 = mat1->cols;
    int cols2 = mat2->cols;
    allocate_matrix(&mat2T, mat2->cols, mat2->rows);
    for(int i = 0; i < mat2->rows; i++){
        for(int j = 0; j < cols2; j++){
            (mat2T->data)[j*cols1 + i] = (mat2->data)[i*cols2 + j];
	}
    }
    double *d2 = mat2T->data;
    double *sumparts;
    __m256d simd_sum;
    __m256d a;
    __m256d b;
#pragma omp parallel for private(sum, a, b, simd_sum, sumparts)
    for(int i = 0; i < result->rows; i++){
      for(int j = 0; j < cols2; j++){
	sumparts = malloc(32);
	simd_sum = _mm256_setzero_pd();
	for(int k = 0; k < cols1/16 * 16; k+=16){
            a = _mm256_loadu_pd(d1+i*cols1+k);
	    b = _mm256_loadu_pd(d2+j*cols1+k);
	    simd_sum = _mm256_fmadd_pd(a, b, simd_sum);
	    a = _mm256_loadu_pd(d1+i*cols1+k+4);
            b = _mm256_loadu_pd(d2+j*cols1+k+4);
            simd_sum = _mm256_fmadd_pd(a, b, simd_sum);
	    a = _mm256_loadu_pd(d1+i*cols1+k+8);
            b = _mm256_loadu_pd(d2+j*cols1+k+8);
            simd_sum = _mm256_fmadd_pd(a, b, simd_sum);
	    a = _mm256_loadu_pd(d1+i*cols1+k+12);
            b = _mm256_loadu_pd(d2+j*cols1+k+12);
            simd_sum = _mm256_fmadd_pd(a, b, simd_sum);
	}
	_mm256_storeu_pd(sumparts, simd_sum);
	sum = sumparts[0] + sumparts[1] + sumparts[2] + sumparts[3];
	free(sumparts);
	for(int k = cols1/16 * 16; k < cols1; k++){
	  sum += d1[i*cols1 + k] * d2[j*cols1 + k];
	}
	rd[i*cols2 + j] = sum;
      }
    }
    deallocate_matrix(mat2T);
    return 0;    
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */
    matrix *mat2;
    matrix *mat3;
    allocate_matrix(&mat2, mat->rows, mat->cols);
    allocate_matrix(&mat3, mat->rows, mat->cols);
    for(int k = 0; k < mat->rows; k++){
      set(mat3, k, k, 1);
    }
    double *tmp;
    if(pow>3){
    	mul_matrix(result, mat, mat);
	mul_matrix(mat2, result, result);
	for(int i = 0; i < pow/4; i++){
	    mul_matrix(result, mat2, mat3);
	    tmp = mat3->data;
	    mat3->data = result->data;
	    result->data = tmp;
	}
    }
    for(int i = 0; i < pow % 4; i++){
	mul_matrix(result, mat3, mat);
	tmp = result->data;
	result->data = mat3->data;
	mat3->data = tmp;
    }
    tmp = result->data;
    result->data = mat3->data;
    mat3->data = tmp;
    deallocate_matrix(mat2);
    deallocate_matrix(mat3);
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    int size = (result->rows)*(result->cols);
    double *rd = result->data;
    double *md = mat->data;
    __m256d a;
    __m256d zero = _mm256_setzero_pd();
#pragma omp parallel for private(a)
    for(int i = 0; i < size/16 * 16; i+=16){
      a = _mm256_loadu_pd(md+i);
      _mm256_storeu_pd(rd + i, _mm256_sub_pd(zero, a));
      a = _mm256_loadu_pd(md+i+4);
      _mm256_storeu_pd(rd + i+4, _mm256_sub_pd(zero, a));
      a = _mm256_loadu_pd(md+i+8);
      _mm256_storeu_pd(rd + i+8, _mm256_sub_pd(zero, a));
      a = _mm256_loadu_pd(md+i+12);
      _mm256_storeu_pd(rd + i+12, _mm256_sub_pd(zero, a));
    }
    for(int i = size/16 * 16; i < size; i++){
        rd[i] = -md[i];
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    int size = (result->rows)*(result->cols);
    double *rd = result->data;
    double *md = mat->data;
#pragma omp parallel for
	for(int i = 0; i < size/4 * 4; i+=4){
		rd[i] = (md[i] < 0)? -md[i] : md[i];
      		rd[i+1] = (md[i+1] < 0)? -md[i+1] : md[i+1];
      		rd[i+2] = (md[i+2] < 0)? -md[i+2] : md[i+2];
      		rd[i+3] = (md[i+3] < 0)? -md[i+3] : md[i+3];
    	}
    for(int i = size/4 * 4; i < size; i++){
        rd[i] = (md[i] < 0)? -md[i] : md[i];
    }
    return 0;
}

