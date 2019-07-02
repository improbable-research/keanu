package io.improbable.keanu.tensor.dbl;

import io.improbable.keanu.tensor.buffer.JVMBuffer;
import io.improbable.keanu.tensor.buffer.PrimitiveNumberWrapper;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

public class DoubleBuffer {

    public static final class DoubleArrayWrapperFactory implements JVMBuffer.PrimitiveNumberWrapperFactory<Double, PrimitiveDoubleWrapper> {

        @Override
        public final PrimitiveDoubleWrapper createNew(final int size) {
            if (size == 1) {
                return new DoubleWrapper(0);
            } else {
                return new DoubleArrayWrapper(new double[size]);
            }
        }

        @Override
        public PrimitiveDoubleWrapper createNew(Double value) {
            return new DoubleWrapper(value);
        }

        @Override
        public PrimitiveDoubleWrapper zeroes(final int size) {
            return createNew(size);
        }

        @Override
        public PrimitiveDoubleWrapper ones(final int size) {
            double[] ones = new double[size];
            Arrays.fill(ones, 1.0);
            return create(ones);
        }

        public final PrimitiveDoubleWrapper create(final double[] data) {
            if (data.length == 1) {
                return new DoubleWrapper(data[0]);
            } else {
                return new DoubleArrayWrapper(data);
            }
        }
    }

    public interface PrimitiveDoubleWrapper extends PrimitiveNumberWrapper<Double, PrimitiveDoubleWrapper> {
    }

    public static final class DoubleArrayWrapper implements PrimitiveDoubleWrapper {

        private final double[] array;

        public DoubleArrayWrapper(final double[] array) {
            this.array = array;
        }

        @Override
        public Double get(final int index) {
            return array[index];
        }

        @Override
        public DoubleArrayWrapper set(final Double value, final int index) {
            array[index] = value;
            return this;
        }

        @Override
        public int getLength() {
            return array.length;
        }

        @Override
        public PrimitiveDoubleWrapper copy() {
            return new DoubleArrayWrapper(Arrays.copyOf(array, array.length));
        }

        @Override
        public DoubleArrayWrapper copyFrom(JVMBuffer.PrimitiveArrayWrapper<Double, ?> src, int srcPos, int destPos, int length) {
            if (src instanceof DoubleArrayWrapper) {
                System.arraycopy(((DoubleArrayWrapper) src).array, srcPos, array, destPos, length);
            } else {
                for (int i = 0; i < length; i++) {
                    array[destPos + i] = src.get(srcPos + i);
                }
            }
            return this;
        }

        @Override
        public Double sum() {
            double result = 0;
            for (int i = 0; i < array.length; i++) {
                result += array[i];
            }
            return result;
        }

        @Override
        public Double product() {
            double result = 1.0;
            for (int i = 0; i < array.length; i++) {
                result *= array[i];
            }
            return result;
        }

        @Override
        public DoubleArrayWrapper times(Double that) {
            for (int i = 0; i < array.length; i++) {
                array[i] *= that;
            }
            return this;
        }

        @Override
        public DoubleArrayWrapper div(Double that) {
            for (int i = 0; i < array.length; i++) {
                array[i] /= that;
            }
            return this;
        }

        @Override
        public DoubleArrayWrapper plus(Double that) {
            for (int i = 0; i < array.length; i++) {
                array[i] += that;
            }
            return this;
        }

        @Override
        public DoubleArrayWrapper plus(int index, Double that) {
            array[index] += that;
            return this;
        }

        @Override
        public DoubleArrayWrapper times(int index, Double that) {
            array[index] *= that;
            return this;
        }

        @Override
        public DoubleArrayWrapper minus(Double that) {
            for (int i = 0; i < array.length; i++) {
                array[i] -= that;
            }
            return this;
        }

        @Override
        public DoubleArrayWrapper pow(Double that) {
            for (int i = 0; i < array.length; i++) {
                array[i] = FastMath.pow(array[i], that);
            }
            return this;
        }

        @Override
        public DoubleArrayWrapper reverseDiv(Double that) {
            for (int i = 0; i < array.length; i++) {
                array[i] = that / array[i];
            }
            return this;
        }

        @Override
        public DoubleArrayWrapper reverseMinus(Double that) {
            for (int i = 0; i < array.length; i++) {
                array[i] = that - array[i];
            }
            return this;
        }

        @Override
        public DoubleArrayWrapper applyRight(BiFunction<Double, Double, Double> mapper, Double rightArg) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(array[i], rightArg);
            }
            return this;
        }

        @Override
        public DoubleArrayWrapper applyLeft(BiFunction<Double, Double, Double> mapper, Double leftArg) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(leftArg, array[i]);
            }
            return this;
        }

        @Override
        public DoubleArrayWrapper apply(Function<Double, Double> mapper) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(array[i]);
            }
            return this;
        }

        @Override
        public int[] asIntegerArray() {

            int[] intBuffer = new int[array.length];
            for (int i = 0; i < array.length; i++) {
                intBuffer[i] = (int) array[i];
            }
            return intBuffer;
        }

        @Override
        public double[] asDoubleArray() {
            return array;
        }

        @Override
        public Double[] asArray() {
            return ArrayUtils.toObject(array);
        }

        public boolean equals(final Object o) {
            if (o == this) return true;
            if (!(o instanceof PrimitiveDoubleWrapper)) return false;
            final PrimitiveDoubleWrapper other = (PrimitiveDoubleWrapper) o;
            if (!Arrays.equals(this.array, other.asDoubleArray())) return false;
            return true;
        }

        public int hashCode() {
            final int PRIME = 59;
            int result = 1;
            result = result * PRIME + Arrays.hashCode(this.array);
            return result;
        }
    }

    public static final class DoubleWrapper extends JVMBuffer.SingleValueWrapper<Double, PrimitiveDoubleWrapper> implements PrimitiveDoubleWrapper {

        public DoubleWrapper(final double value) {
            super(value);
        }

        @Override
        public Double sum() {
            return value;
        }

        @Override
        public Double product() {
            return value;
        }

        @Override
        public PrimitiveDoubleWrapper times(Double that) {
            value *= that;
            return this;
        }

        @Override
        public PrimitiveDoubleWrapper div(Double that) {
            value /= that;
            return this;
        }

        @Override
        public PrimitiveDoubleWrapper plus(Double that) {
            value += that;
            return this;
        }

        @Override
        public PrimitiveDoubleWrapper plus(int index, Double that) {
            value += that;
            return this;
        }

        @Override
        public PrimitiveDoubleWrapper minus(Double that) {
            value -= that;
            return this;
        }

        @Override
        public PrimitiveDoubleWrapper pow(Double that) {
            value = FastMath.pow(value, that);
            return this;
        }

        @Override
        public PrimitiveDoubleWrapper reverseDiv(Double that) {
            value = that / value;
            return this;
        }

        @Override
        public PrimitiveDoubleWrapper reverseMinus(Double that) {
            value = that - value;
            return this;
        }

        @Override
        public PrimitiveDoubleWrapper times(int index, Double that) {
            value *= that;
            return this;
        }

        @Override
        public int[] asIntegerArray() {
            return new int[]{value.intValue()};
        }

        @Override
        public double[] asDoubleArray() {
            return new double[]{value};
        }

        @Override
        public Double[] asArray() {
            return new Double[]{value};
        }

        @Override
        public PrimitiveDoubleWrapper copy() {
            return new DoubleWrapper(value);
        }

        @Override
        protected PrimitiveDoubleWrapper getThis() {
            return this;
        }
    }
}
