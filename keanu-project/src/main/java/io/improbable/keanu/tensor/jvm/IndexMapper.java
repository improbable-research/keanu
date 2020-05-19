package io.improbable.keanu.tensor.jvm;

public interface IndexMapper {

    long[] getResultShape();

    long[] getResultStride();

    /**
     * @param resultIndex the index in the result buffer
     * @return the index in the source buffer that maps to the result buffer.
     */
    long getSourceIndexFromResultIndex(long resultIndex);
}
