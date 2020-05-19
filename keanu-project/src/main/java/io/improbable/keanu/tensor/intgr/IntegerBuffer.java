package io.improbable.keanu.tensor.intgr;

import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.bool.BooleanBuffer;
import io.improbable.keanu.tensor.jvm.buffer.JVMBuffer;
import io.improbable.keanu.tensor.jvm.buffer.PrimitiveNumberWrapper;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

public class IntegerBuffer {

    public static final class IntegerArrayWrapperFactory implements JVMBuffer.PrimitiveNumberWrapperFactory<Integer, PrimitiveIntegerWrapper> {

        @Override
        public final PrimitiveIntegerWrapper createNew(final long size) {
            if (size == 1) {
                return new IntegerWrapper(0);
            } else {
                return new IntegerArrayWrapper(new int[Ints.checkedCast(size)]);
            }
        }

        @Override
        public PrimitiveIntegerWrapper createNew(Integer value) {
            return new IntegerWrapper(value);
        }

        @Override
        public PrimitiveIntegerWrapper zeroes(final long size) {
            return createNew(size);
        }

        @Override
        public PrimitiveIntegerWrapper ones(final long size) {
            if (size == 1) {
                return new IntegerWrapper(1);
            } else {
                int[] ones = new int[Ints.checkedCast(size)];
                Arrays.fill(ones, 1);
                return new IntegerArrayWrapper(ones);
            }
        }

        public final PrimitiveIntegerWrapper create(final int[] data) {
            if (data.length == 1) {
                return new IntegerWrapper(data[0]);
            } else {
                return new IntegerArrayWrapper(data);
            }
        }
    }

    public interface PrimitiveIntegerWrapper extends PrimitiveNumberWrapper<Integer, PrimitiveIntegerWrapper> {
        @Override
        Integer[] asArray();
    }

    public static final class IntegerArrayWrapper implements PrimitiveIntegerWrapper {

        private final int[] array;

        public IntegerArrayWrapper(final int[] array) {
            this.array = array;
        }

        @Override
        public Integer get(final long index) {
            return array[Ints.checkedCast(index)];
        }

        @Override
        public IntegerArrayWrapper set(final Integer value, final long index) {
            array[Ints.checkedCast(index)] = value;
            return this;
        }

        @Override
        public long getLength() {
            return array.length;
        }

        @Override
        public PrimitiveIntegerWrapper copy() {
            return new IntegerArrayWrapper(Arrays.copyOf(array, array.length));
        }

        @Override
        public IntegerArrayWrapper copyFrom(JVMBuffer.PrimitiveArrayWrapper<Integer, ?> src, long srcPos, long destPos, long length) {
            if (src instanceof IntegerArrayWrapper) {
                System.arraycopy(((IntegerArrayWrapper) src).array, Ints.checkedCast(srcPos), array, Ints.checkedCast(destPos), Ints.checkedCast(length));
            } else {
                for (int i = 0; i < length; i++) {
                    array[Ints.checkedCast(destPos + i)] = src.get(srcPos + i);
                }
            }
            return this;
        }

        @Override
        public Integer sum() {
            int result = 0;
            for (int i = 0; i < array.length; i++) {
                result += array[i];
            }
            return result;
        }

        @Override
        public Integer product() {
            int result = 1;
            for (int i = 0; i < array.length; i++) {
                result *= array[i];
            }
            return result;
        }

        @Override
        public Integer max() {
            int result = Integer.MIN_VALUE;
            for (int i = 0; i < array.length; i++) {
                result = Math.max(array[i], result);
            }
            return result;
        }

        @Override
        public Integer min() {
            int result = Integer.MAX_VALUE;
            for (int i = 0; i < array.length; i++) {
                result = Math.min(array[i], result);
            }
            return result;
        }

        @Override
        public IntegerArrayWrapper times(Integer that) {
            for (int i = 0; i < array.length; i++) {
                array[i] *= that;
            }
            return this;
        }

        @Override
        public IntegerArrayWrapper div(Integer that) {
            for (int i = 0; i < array.length; i++) {
                array[i] /= that;
            }
            return this;
        }

        @Override
        public IntegerArrayWrapper plus(Integer that) {
            for (int i = 0; i < array.length; i++) {
                array[i] += that;
            }
            return this;
        }

        @Override
        public IntegerArrayWrapper plus(long index, Integer that) {
            array[Ints.checkedCast(index)] += that;
            return this;
        }

        @Override
        public IntegerArrayWrapper times(long index, Integer that) {
            array[Ints.checkedCast(index)] *= that;
            return this;
        }

        @Override
        public IntegerArrayWrapper minus(Integer that) {
            for (int i = 0; i < array.length; i++) {
                array[i] -= that;
            }
            return this;
        }

        @Override
        public IntegerArrayWrapper pow(Integer that) {
            for (int i = 0; i < array.length; i++) {
                array[i] = (int) FastMath.pow(array[i], that);
            }
            return this;
        }

        @Override
        public IntegerArrayWrapper reverseDiv(Integer that) {
            for (int i = 0; i < array.length; i++) {
                array[i] = that / array[i];
            }
            return this;
        }

        @Override
        public IntegerArrayWrapper reverseMinus(Integer that) {
            for (int i = 0; i < array.length; i++) {
                array[i] = that - array[i];
            }
            return this;
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper greaterThan(Integer that) {
            BooleanBuffer.PrimitiveBooleanWrapper boolBuffer = BooleanBuffer.factory.createNew(array.length);
            for (int i = 0; i < array.length; i++) {
                boolBuffer.set(array[i] > that, i);
            }
            return boolBuffer;
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper lessThan(Integer that) {
            BooleanBuffer.PrimitiveBooleanWrapper boolBuffer = BooleanBuffer.factory.createNew(array.length);
            for (int i = 0; i < array.length; i++) {
                boolBuffer.set(array[i] < that, i);
            }
            return boolBuffer;
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper greaterThanOrEqual(Integer that) {
            BooleanBuffer.PrimitiveBooleanWrapper boolBuffer = BooleanBuffer.factory.createNew(array.length);
            for (int i = 0; i < array.length; i++) {
                boolBuffer.set(array[i] >= that, i);
            }
            return boolBuffer;
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper lessThanOrEqual(Integer that) {
            BooleanBuffer.PrimitiveBooleanWrapper boolBuffer = BooleanBuffer.factory.createNew(array.length);
            for (int i = 0; i < array.length; i++) {
                boolBuffer.set(array[i] <= that, i);
            }
            return boolBuffer;
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper equal(Integer that) {
            BooleanBuffer.PrimitiveBooleanWrapper boolBuffer = BooleanBuffer.factory.createNew(array.length);
            for (int i = 0; i < array.length; i++) {
                boolBuffer.set(array[i] == that, i);
            }
            return boolBuffer;
        }

        @Override
        public IntegerArrayWrapper applyRight(BiFunction<Integer, Integer, Integer> mapper, Integer rightArg) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(array[i], rightArg);
            }
            return this;
        }

        @Override
        public IntegerArrayWrapper applyLeft(BiFunction<Integer, Integer, Integer> mapper, Integer leftArg) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(leftArg, array[i]);
            }
            return this;
        }

        @Override
        public IntegerArrayWrapper apply(Function<Integer, Integer> mapper) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(array[i]);
            }
            return this;
        }

        @Override
        public int[] asIntegerArray() {
            return array;
        }

        @Override
        public long[] asLongArray() {
            long[] longBuffer = new long[array.length];
            for (int i = 0; i < array.length; i++) {
                longBuffer[i] = array[i];
            }
            return longBuffer;
        }

        @Override
        public double[] asDoubleArray() {

            double[] dbls = new double[array.length];
            for (int i = 0; i < dbls.length; i++) {
                dbls[i] = array[i];
            }
            return dbls;
        }

        @Override
        public Integer[] asArray() {
            return ArrayUtils.toObject(array);
        }

        public boolean equals(final Object o) {
            if (o == this) return true;
            if (!(o instanceof PrimitiveIntegerWrapper)) return false;
            final PrimitiveIntegerWrapper other = (PrimitiveIntegerWrapper) o;
            if (!Arrays.equals(this.array, other.asIntegerArray())) return false;
            return true;
        }

        public int hashCode() {
            final int PRIME = 59;
            int result = 1;
            result = result * PRIME + Arrays.hashCode(this.array);
            return result;
        }
    }

    public static final class IntegerWrapper extends JVMBuffer.SingleValueWrapper<Integer, PrimitiveIntegerWrapper> implements PrimitiveIntegerWrapper {

        public IntegerWrapper(final int value) {
            super(value);
        }

        @Override
        public Integer sum() {
            return value;
        }

        @Override
        public Integer product() {
            return value;
        }

        @Override
        public Integer max() {
            return value;
        }

        @Override
        public Integer min() {
            return value;
        }

        @Override
        public PrimitiveIntegerWrapper times(Integer that) {
            value *= that;
            return this;
        }

        @Override
        public PrimitiveIntegerWrapper div(Integer that) {
            value /= that;
            return this;
        }

        @Override
        public PrimitiveIntegerWrapper plus(Integer that) {
            value += that;
            return this;
        }

        @Override
        public PrimitiveIntegerWrapper plus(long index, Integer that) {
            value += that;
            return this;
        }

        @Override
        public PrimitiveIntegerWrapper minus(Integer that) {
            value -= that;
            return this;
        }

        @Override
        public PrimitiveIntegerWrapper pow(Integer that) {
            value = (int) FastMath.pow(value, that);
            return this;
        }

        @Override
        public PrimitiveIntegerWrapper reverseDiv(Integer that) {
            value = that / value;
            return this;
        }

        @Override
        public PrimitiveIntegerWrapper reverseMinus(Integer that) {
            value = that - value;
            return this;
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper greaterThan(Integer that) {
            return BooleanBuffer.factory.createNew(value > that);
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper lessThan(Integer that) {
            return BooleanBuffer.factory.createNew(value < that);
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper greaterThanOrEqual(Integer that) {
            return BooleanBuffer.factory.createNew(value >= that);
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper lessThanOrEqual(Integer that) {
            return BooleanBuffer.factory.createNew(value <= that);
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper equal(Integer that) {
            return BooleanBuffer.factory.createNew(value.equals(that));
        }

        @Override
        public PrimitiveIntegerWrapper times(long index, Integer that) {
            value *= that;
            return this;
        }

        @Override
        public int[] asIntegerArray() {
            return new int[]{value};
        }

        @Override
        public long[] asLongArray() {
            return new long[]{value};
        }

        @Override
        public double[] asDoubleArray() {
            return new double[]{value.doubleValue()};
        }

        @Override
        public Integer[] asArray() {
            return new Integer[]{value};
        }

        @Override
        public PrimitiveIntegerWrapper copy() {
            return new IntegerWrapper(value);
        }

        @Override
        protected PrimitiveIntegerWrapper getThis() {
            return this;
        }
    }
}
