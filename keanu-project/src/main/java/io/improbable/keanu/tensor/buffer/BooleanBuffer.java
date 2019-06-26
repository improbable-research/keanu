package io.improbable.keanu.tensor.buffer;

import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

public class BooleanBuffer {

    public static final class BooleanArrayWrapperFactory implements JVMBuffer.ArrayWrapperFactory<Boolean, PrimitiveBooleanWrapper> {

        @Override
        public final BooleanBuffer.PrimitiveBooleanWrapper createNew(final int size) {
            if (size == 1) {
                return new BooleanBuffer.BooleanWrapper(false);
            } else {
                return new BooleanBuffer.BooleanArrayWrapper(new boolean[size]);
            }
        }

        public final BooleanBuffer.PrimitiveBooleanWrapper create(boolean[] data) {
            if (data.length == 1) {
                return new BooleanBuffer.BooleanWrapper(data[0]);
            } else {
                return new BooleanBuffer.BooleanArrayWrapper(data);
            }
        }
    }

    public interface PrimitiveBooleanWrapper extends JVMBuffer.PrimitiveArrayWrapper<Boolean, PrimitiveBooleanWrapper> {

        int[] asIntegerArray();

        double[] asDoubleArray();

        boolean[] asBooleanArray();
    }

    public static final class BooleanArrayWrapper implements PrimitiveBooleanWrapper {

        private final boolean[] array;

        public BooleanArrayWrapper(final boolean[] array) {
            this.array = array;
        }

        @Override
        public Boolean get(final int index) {
            return array[index];
        }

        @Override
        public void set(final Boolean value, final int index) {
            array[index] = value;
        }

        @Override
        public int getLength() {
            return array.length;
        }

        @Override
        public PrimitiveBooleanWrapper copy() {
            return new BooleanBuffer.BooleanArrayWrapper(Arrays.copyOf(array, array.length));
        }

        @Override
        public void copyFrom(JVMBuffer.PrimitiveArrayWrapper<Boolean, ?> src, int srcPos, int destPos, int length) {
            if (src instanceof BooleanArrayWrapper) {
                System.arraycopy(((BooleanArrayWrapper) src).array, srcPos, array, destPos, length);
            } else {
                for (int i = 0; i < length; i++) {
                    array[destPos + i] = src.get(srcPos + i);
                }
            }
        }

        @Override
        public void applyRight(BiFunction<Boolean, Boolean, Boolean> mapper, Boolean rightArg) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(array[i], rightArg);
            }
        }

        @Override
        public void applyLeft(BiFunction<Boolean, Boolean, Boolean> mapper, Boolean leftArg) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(leftArg, array[i]);
            }
        }

        @Override
        public void apply(Function<Boolean, Boolean> mapper) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(array[i]);
            }
        }

        @Override
        public int[] asIntegerArray() {

            int[] intBuffer = new int[array.length];
            for (int i = 0; i < array.length; i++) {
                intBuffer[i] = array[i] ? 1 : 0;
            }
            return intBuffer;
        }

        @Override
        public double[] asDoubleArray() {

            double[] intBuffer = new double[array.length];
            for (int i = 0; i < array.length; i++) {
                intBuffer[i] = array[i] ? 1.0 : 0.0;
            }
            return intBuffer;
        }

        @Override
        public boolean[] asBooleanArray() {
            return array;
        }

        @Override
        public Boolean[] asArray() {
            return ArrayUtils.toObject(array);
        }

        public boolean equals(final Object o) {
            if (o == this) return true;
            if (!(o instanceof BooleanBuffer.PrimitiveBooleanWrapper)) return false;
            final BooleanBuffer.PrimitiveBooleanWrapper other = (BooleanBuffer.PrimitiveBooleanWrapper) o;
            if (!Arrays.equals(this.array, other.asBooleanArray())) return false;
            return true;
        }

        public int hashCode() {
            final int PRIME = 59;
            int result = 1;
            result = result * PRIME + Arrays.hashCode(this.array);
            return result;
        }
    }

    public static final class BooleanWrapper extends JVMBuffer.SingleValueWrapper<Boolean, PrimitiveBooleanWrapper> implements PrimitiveBooleanWrapper {

        public BooleanWrapper(final boolean value) {
            super(value);
        }

        @Override
        public PrimitiveBooleanWrapper copy() {
            return new BooleanWrapper(value);
        }

        @Override
        public int[] asIntegerArray() {
            return new int[]{value ? 1 : 0};
        }

        @Override
        public double[] asDoubleArray() {
            return new double[]{value ? 1.0 : 0.0};
        }

        @Override
        public boolean[] asBooleanArray() {
            return new boolean[]{value};
        }

        @Override
        public Boolean[] asArray() {
            return new Boolean[]{value};
        }
    }
}
