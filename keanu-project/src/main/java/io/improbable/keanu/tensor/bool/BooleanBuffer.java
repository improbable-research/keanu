package io.improbable.keanu.tensor.bool;

import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.jvm.buffer.JVMBuffer;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

public class BooleanBuffer {

    public static final BooleanArrayWrapperFactory factory = new BooleanArrayWrapperFactory();

    public static final class BooleanArrayWrapperFactory implements JVMBuffer.ArrayWrapperFactory<Boolean, PrimitiveBooleanWrapper> {

        @Override
        public final BooleanBuffer.PrimitiveBooleanWrapper createNew(final long size) {
            if (size == 1) {
                return new BooleanBuffer.BooleanWrapper(false);
            } else {
                return new BooleanBuffer.BooleanArrayWrapper(new boolean[Ints.checkedCast(size)]);
            }
        }

        @Override
        public PrimitiveBooleanWrapper createNew(Boolean value) {
            return new BooleanBuffer.BooleanWrapper(value);
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
        public Boolean get(final long index) {
            return array[Ints.checkedCast(index)];
        }

        @Override
        public BooleanArrayWrapper set(final Boolean value, final long index) {
            array[Ints.checkedCast(index)] = value;
            return this;
        }

        @Override
        public long getLength() {
            return array.length;
        }

        @Override
        public PrimitiveBooleanWrapper copy() {
            return new BooleanBuffer.BooleanArrayWrapper(Arrays.copyOf(array, array.length));
        }

        @Override
        public BooleanArrayWrapper copyFrom(JVMBuffer.PrimitiveArrayWrapper<Boolean, ?> src, long srcPos, long destPos, long length) {
            if (src instanceof BooleanArrayWrapper) {
                System.arraycopy(((BooleanArrayWrapper) src).array, Ints.checkedCast(srcPos), array, Ints.checkedCast(destPos), Ints.checkedCast(length));
            } else {
                for (int i = 0; i < length; i++) {
                    array[Ints.checkedCast(destPos + i)] = src.get(srcPos + i);
                }
            }
            return this;
        }

        @Override
        public BooleanArrayWrapper applyRight(BiFunction<Boolean, Boolean, Boolean> mapper, Boolean rightArg) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(array[i], rightArg);
            }
            return this;
        }

        @Override
        public BooleanArrayWrapper applyLeft(BiFunction<Boolean, Boolean, Boolean> mapper, Boolean leftArg) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(leftArg, array[i]);
            }
            return this;
        }

        @Override
        public BooleanArrayWrapper apply(Function<Boolean, Boolean> mapper) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(array[i]);
            }
            return this;
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

        @Override
        protected PrimitiveBooleanWrapper getThis() {
            return this;
        }
    }
}
