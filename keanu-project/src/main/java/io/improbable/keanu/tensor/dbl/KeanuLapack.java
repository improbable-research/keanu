package io.improbable.keanu.tensor.dbl;

import lombok.AllArgsConstructor;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

import static org.bytedeco.openblas.global.openblas.LAPACKE_dgetrf;
import static org.bytedeco.openblas.global.openblas.LAPACKE_dgetri;
import static org.bytedeco.openblas.global.openblas.LAPACKE_dpotrf;
import static org.bytedeco.openblas.global.openblas.LAPACKE_dpotri;
import static org.bytedeco.openblas.global.openblas.LAPACK_ROW_MAJOR;


/**
 * https://software.intel.com/en-us/mkl-developer-reference-c-matrix-factorization-lapack-computational-routines
 *
 * http://www.icl.utk.edu/~mgates3/docs/lapack.html
 * <p>
 * This is a wrapper around a few LAPACK functions used directly in Keanu
 */
public class KeanuLapack {

    @AllArgsConstructor
    public enum Triangular {

        UPPER((byte) 'U'), LOWER((byte) 'L');

        private final byte lapackChar;
    }

    /**
     * https://software.intel.com/en-us/mkl-developer-reference-c-potrf
     * <p>
     * Computes the Cholesky factorization of a symmetric (Hermitian) positive-definite matrix.
     *
     * @param upperLower either upper triangular or lower triangular specifies format of result
     * @param N          length of one dimension in the square matrix to operate on
     * @param buffer     matrix to take cholesky decoposition of and buffer to store result
     * @return 0 if successful, -i if error at i index, or positive if matrix is not positive-definite
     */
    public static int dpotrf(Triangular upperLower, int N, double[] buffer) {
        return dpotrf(upperLower, N, DoubleBuffer.wrap(buffer));
    }

    public static int dpotrf(Triangular upperLower, int N, DoubleBuffer buffer) {
        return LAPACKE_dpotrf(LAPACK_ROW_MAJOR, upperLower.lapackChar, N, buffer, N);
    }

    /**
     * https://software.intel.com/en-us/mkl-developer-reference-c-potri
     * <p>
     * Computes the inverse byt using the Cholesky factorization of a symmetric (Hermitian) positive-definite matrix.
     *
     * @param upperLower either upper triangular or lower triangular specifies format of result
     * @param N          length of one dimension in the square matrix to operate on
     * @param buffer     The cholesky decoposition of and buffer to store result
     * @return 0 if successful, -i if error at i index, or positive if matrix is not positive-definite
     */
    public static int dpotri(Triangular upperLower, int N, double[] buffer) {
        return dpotri(upperLower, N, DoubleBuffer.wrap(buffer));
    }

    public static int dpotri(Triangular upperLower, int N, DoubleBuffer buffer) {
        return LAPACKE_dpotri(LAPACK_ROW_MAJOR, upperLower.lapackChar, N, buffer, N);
    }

    /**
     * https://software.intel.com/en-us/mkl-developer-reference-c-getrf
     * <p>
     * Computes the LU factorization of a general m-by-n matrix.
     *
     * @param m      row dimension of matrix
     * @param n      column dimensino of matrix
     * @param buffer buffer that represents the matrix for LU factorization
     * @param ipiv   The pivot indices
     * @return 0 if successful, -i if error at i index, or i if matrix is singular and ii index is zero
     */
    public static int dgetrf(int m, int n, double[] buffer, int[] ipiv) {
        return dgetrf(m, n, DoubleBuffer.wrap(buffer), IntBuffer.wrap(ipiv));
    }

    public static int dgetrf(int m, int n, DoubleBuffer buffer, IntBuffer ipiv) {
        return LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, buffer, m, ipiv);
    }

    /**
     * https://software.intel.com/en-us/mkl-developer-reference-c-getri
     * <p>
     * Computes the inverse of an LU-factored general matrix.
     *
     * @param N      length of one dimension in the square matrix to operate on
     * @param buffer buffer of data that represents square matrix
     * @param ipiv   The pivot indices
     * @return 0 if successful, -i if error at i index, or i if matrix is singular and ii index is zero
     */
    public static int dgetri(int N, double[] buffer, int[] ipiv) {
        return dgetri(N, DoubleBuffer.wrap(buffer), IntBuffer.wrap(ipiv));
    }

    public static int dgetri(int N, DoubleBuffer buffer, IntBuffer ipiv) {
        return LAPACKE_dgetri(LAPACK_ROW_MAJOR, N, buffer, N, ipiv);
    }
}
