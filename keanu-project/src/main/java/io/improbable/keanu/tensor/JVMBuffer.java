package io.improbable.keanu.tensor;

import java.util.Arrays;
import java.util.function.Function;

public class JVMBuffer {

    public interface PrimitiveArrayWrapper<T> {
        T get(int index);

        void set(T value, int index);

        int getLength();

        PrimitiveArrayWrapper<T> copy();

        void apply(Function<T, T> mapper);
    }

    public interface PrimitiveDoubleWrapper extends PrimitiveArrayWrapper<Double> {

        int[] asIntegerArray();

        double[] asDoubleArray();

        @Override
        PrimitiveDoubleWrapper copy();
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
        public void set(final Double value, final int index) {
            array[index] = value;
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
        public void apply(Function<Double, Double> mapper) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(array[i]);
            }
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

    public static final class DoubleWrapper implements PrimitiveDoubleWrapper {

        private double value;

        public DoubleWrapper(final double value) {
            this.value = value;
        }

        @Override
        public Double get(final int index) {
            return value;
        }

        @Override
        public void set(final Double value, final int index) {
            this.value = value;
        }

        @Override
        public int getLength() {
            return 1;
        }

        @Override
        public PrimitiveDoubleWrapper copy() {
            return new DoubleWrapper(value);
        }

        @Override
        public void apply(Function<Double, Double> mapper) {
            value = mapper.apply(value);
        }

        @Override
        public int[] asIntegerArray() {
            return new int[]{(int) value};
        }

        @Override
        public double[] asDoubleArray() {
            return new double[]{value};
        }

        public boolean equals(final Object o) {
            if (o == this) return true;
            if (!(o instanceof PrimitiveDoubleWrapper)) return false;
            final PrimitiveDoubleWrapper other = (PrimitiveDoubleWrapper) o;
            if (other.getLength() != 1) return false;
            if (Double.compare(this.value, other.get(0)) != 0) return false;
            return true;
        }

        public int hashCode() {
            final int PRIME = 59;
            int result = 1;
            final long $value = Double.doubleToLongBits(this.value);
            result = result * PRIME + (int) ($value >>> 32 ^ $value);
            return result;
        }
    }

    public interface PrimitiveIntegerWrapper extends PrimitiveArrayWrapper<Integer> {
    }

    public static final class IntegerArrayWrapper implements PrimitiveIntegerWrapper {

        private final int[] array;

        public IntegerArrayWrapper(final int[] array) {
            this.array = array;
        }

        @Override
        public Integer get(final int index) {
            return array[index];
        }

        @Override
        public void set(final Integer value, final int index) {
            array[index] = value;
        }

        @Override
        public int getLength() {
            return array.length;
        }

        @Override
        public PrimitiveArrayWrapper<Integer> copy() {
            return new IntegerArrayWrapper(Arrays.copyOf(array, array.length));
        }

        @Override
        public void apply(Function<Integer, Integer> mapper) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(array[i]);
            }
        }
    }

    public static final class IntegerWrapper implements PrimitiveIntegerWrapper {

        private int value;

        public IntegerWrapper(final int value) {
            this.value = value;
        }

        @Override
        public Integer get(final int index) {
            return value;
        }

        @Override
        public void set(final Integer value, final int index) {
            this.value = value;
        }

        @Override
        public int getLength() {
            return 1;
        }

        @Override
        public PrimitiveArrayWrapper<Integer> copy() {
            return new IntegerWrapper(value);
        }

        @Override
        public void apply(Function<Integer, Integer> mapper) {
            value = mapper.apply(value);
        }
    }

    public interface ArrayWrapperFactory<D, T extends PrimitiveArrayWrapper<D>> {
        T createNew(int size);
    }

    public static final class DoubleArrayWrapperFactory implements ArrayWrapperFactory<Double, PrimitiveDoubleWrapper> {

        @Override
        public final PrimitiveDoubleWrapper createNew(final int size) {
            if (size == 1) {
                return new DoubleWrapper(0);
            } else {
                return new DoubleArrayWrapper(new double[size]);
            }
        }

        public final PrimitiveDoubleWrapper create(double[] data) {
            if (data.length == 1) {
                return new DoubleWrapper(data[0]);
            } else {
                return new DoubleArrayWrapper(data);
            }
        }
    }

    public static final class IntegerArrayWrapperFactory implements ArrayWrapperFactory<Integer, PrimitiveArrayWrapper<Integer>> {

        @Override
        public final PrimitiveArrayWrapper<Integer> createNew(final int size) {
            if (size == 1) {
                return new IntegerWrapper(0);
            } else {
                return new IntegerArrayWrapper(new int[size]);
            }
        }
    }
}
