package io.improbable.keanu.tensor.lng;

import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.bool.BooleanBuffer;
import io.improbable.keanu.tensor.jvm.buffer.JVMBuffer;
import io.improbable.keanu.tensor.jvm.buffer.PrimitiveNumberWrapper;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

public class LongBuffer {

    public static final class LongArrayWrapperFactory implements JVMBuffer.PrimitiveNumberWrapperFactory<Long, PrimitiveLongWrapper> {

        @Override
        public final PrimitiveLongWrapper createNew(final long size) {
            if (size == 1) {
                return new LongWrapper(0);
            } else {
                return new LongArrayWrapper(new long[Ints.checkedCast(size)]);
            }
        }

        @Override
        public PrimitiveLongWrapper createNew(Long value) {
            return new LongWrapper(value);
        }

        @Override
        public PrimitiveLongWrapper zeroes(final long size) {
            return createNew(size);
        }

        @Override
        public PrimitiveLongWrapper ones(final long size) {
            if (size == 1) {
                return new LongWrapper(1);
            } else {
                long[] ones = new long[Ints.checkedCast(size)];
                Arrays.fill(ones, 1);
                return new LongArrayWrapper(ones);
            }
        }

        public final PrimitiveLongWrapper create(final long[] data) {
            if (data.length == 1) {
                return new LongWrapper(data[0]);
            } else {
                return new LongArrayWrapper(data);
            }
        }
    }

    public interface PrimitiveLongWrapper extends PrimitiveNumberWrapper<Long, PrimitiveLongWrapper> {
        @Override
        Long[] asArray();
    }

    public static final class LongArrayWrapper implements PrimitiveLongWrapper {

        private final long[] array;

        public LongArrayWrapper(final long[] array) {
            this.array = array;
        }

        @Override
        public Long get(final long index) {
            return array[Ints.checkedCast(index)];
        }

        @Override
        public LongArrayWrapper set(final Long value, final long index) {
            array[Ints.checkedCast(index)] = value;
            return this;
        }

        @Override
        public long getLength() {
            return array.length;
        }

        @Override
        public PrimitiveLongWrapper copy() {
            return new LongArrayWrapper(Arrays.copyOf(array, array.length));
        }

        @Override
        public LongArrayWrapper copyFrom(JVMBuffer.PrimitiveArrayWrapper<Long, ?> src, long srcPos, long destPos, long length) {
            if (src instanceof LongArrayWrapper) {
                System.arraycopy(((LongArrayWrapper) src).array, Ints.checkedCast(srcPos), array, Ints.checkedCast(destPos), Ints.checkedCast(length));
            } else {
                for (int i = 0; i < length; i++) {
                    array[Ints.checkedCast(destPos + i)] = src.get(srcPos + i);
                }
            }
            return this;
        }

        @Override
        public Long sum() {
            long result = 0;
            for (int i = 0; i < array.length; i++) {
                result += array[i];
            }
            return result;
        }

        @Override
        public Long product() {
            long result = 1;
            for (int i = 0; i < array.length; i++) {
                result *= array[i];
            }
            return result;
        }

        @Override
        public LongArrayWrapper times(Long that) {
            for (int i = 0; i < array.length; i++) {
                array[i] *= that;
            }
            return this;
        }

        @Override
        public LongArrayWrapper div(Long that) {
            for (int i = 0; i < array.length; i++) {
                array[i] /= that;
            }
            return this;
        }

        @Override
        public LongArrayWrapper plus(Long that) {
            for (int i = 0; i < array.length; i++) {
                array[i] += that;
            }
            return this;
        }

        @Override
        public LongArrayWrapper plus(long index, Long that) {
            array[Ints.checkedCast(index)] += that;
            return this;
        }

        @Override
        public LongArrayWrapper times(long index, Long that) {
            array[Ints.checkedCast(index)] *= that;
            return this;
        }

        @Override
        public LongArrayWrapper minus(Long that) {
            for (int i = 0; i < array.length; i++) {
                array[i] -= that;
            }
            return this;
        }

        @Override
        public LongArrayWrapper pow(Long that) {
            for (int i = 0; i < array.length; i++) {
                array[i] = (long) FastMath.pow(array[i], that);
            }
            return this;
        }

        @Override
        public LongArrayWrapper reverseDiv(Long that) {
            for (int i = 0; i < array.length; i++) {
                array[i] = that / array[i];
            }
            return this;
        }

        @Override
        public LongArrayWrapper reverseMinus(Long that) {
            for (int i = 0; i < array.length; i++) {
                array[i] = that - array[i];
            }
            return this;
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper greaterThan(Long that) {
            BooleanBuffer.PrimitiveBooleanWrapper boolBuffer = BooleanBuffer.factory.createNew(array.length);
            for (int i = 0; i < array.length; i++) {
                boolBuffer.set(array[i] > that, i);
            }
            return boolBuffer;
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper lessThan(Long that) {
            BooleanBuffer.PrimitiveBooleanWrapper boolBuffer = BooleanBuffer.factory.createNew(array.length);
            for (int i = 0; i < array.length; i++) {
                boolBuffer.set(array[i] < that, i);
            }
            return boolBuffer;
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper greaterThanOrEqual(Long that) {
            BooleanBuffer.PrimitiveBooleanWrapper boolBuffer = BooleanBuffer.factory.createNew(array.length);
            for (int i = 0; i < array.length; i++) {
                boolBuffer.set(array[i] >= that, i);
            }
            return boolBuffer;
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper lessThanOrEqual(Long that) {
            BooleanBuffer.PrimitiveBooleanWrapper boolBuffer = BooleanBuffer.factory.createNew(array.length);
            for (int i = 0; i < array.length; i++) {
                boolBuffer.set(array[i] <= that, i);
            }
            return boolBuffer;
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper equal(Long that) {
            BooleanBuffer.PrimitiveBooleanWrapper boolBuffer = BooleanBuffer.factory.createNew(array.length);
            for (int i = 0; i < array.length; i++) {
                boolBuffer.set(array[i] == that, i);
            }
            return boolBuffer;
        }

        @Override
        public LongArrayWrapper applyRight(BiFunction<Long, Long, Long> mapper, Long rightArg) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(array[i], rightArg);
            }
            return this;
        }

        @Override
        public LongArrayWrapper applyLeft(BiFunction<Long, Long, Long> mapper, Long leftArg) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(leftArg, array[i]);
            }
            return this;
        }

        @Override
        public LongArrayWrapper apply(Function<Long, Long> mapper) {
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
        public long[] asLongArray() {
            return array;
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
        public Long[] asArray() {
            return ArrayUtils.toObject(array);
        }

        public boolean equals(final Object o) {
            if (o == this) return true;
            if (!(o instanceof PrimitiveLongWrapper)) return false;
            final PrimitiveLongWrapper other = (PrimitiveLongWrapper) o;
            if (!Arrays.equals(this.array, other.asLongArray())) return false;
            return true;
        }

        public int hashCode() {
            final int PRIME = 59;
            int result = 1;
            result = result * PRIME + Arrays.hashCode(this.array);
            return result;
        }
    }

    public static final class LongWrapper extends JVMBuffer.SingleValueWrapper<Long, PrimitiveLongWrapper> implements PrimitiveLongWrapper {

        public LongWrapper(final long value) {
            super(value);
        }

        @Override
        public Long sum() {
            return value;
        }

        @Override
        public Long product() {
            return value;
        }

        @Override
        public PrimitiveLongWrapper times(Long that) {
            value *= that;
            return this;
        }

        @Override
        public PrimitiveLongWrapper div(Long that) {
            value /= that;
            return this;
        }

        @Override
        public PrimitiveLongWrapper plus(Long that) {
            value += that;
            return this;
        }

        @Override
        public PrimitiveLongWrapper plus(long index, Long that) {
            value += that;
            return this;
        }

        @Override
        public PrimitiveLongWrapper minus(Long that) {
            value -= that;
            return this;
        }

        @Override
        public PrimitiveLongWrapper pow(Long that) {
            value = (long) FastMath.pow(value, that);
            return this;
        }

        @Override
        public PrimitiveLongWrapper reverseDiv(Long that) {
            value = that / value;
            return this;
        }

        @Override
        public PrimitiveLongWrapper reverseMinus(Long that) {
            value = that - value;
            return this;
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper greaterThan(Long that) {
            return BooleanBuffer.factory.createNew(value > that);
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper lessThan(Long that) {
            return BooleanBuffer.factory.createNew(value < that);
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper greaterThanOrEqual(Long that) {
            return BooleanBuffer.factory.createNew(value >= that);
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper lessThanOrEqual(Long that) {
            return BooleanBuffer.factory.createNew(value <= that);
        }

        @Override
        public BooleanBuffer.PrimitiveBooleanWrapper equal(Long that) {
            return BooleanBuffer.factory.createNew(value.equals(that));
        }

        @Override
        public PrimitiveLongWrapper times(long index, Long that) {
            value *= that;
            return this;
        }

        @Override
        public int[] asIntegerArray() {
            return new int[]{value.intValue()};
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
        public Long[] asArray() {
            return new Long[]{value};
        }

        @Override
        public PrimitiveLongWrapper copy() {
            return new LongWrapper(value);
        }

        @Override
        protected PrimitiveLongWrapper getThis() {
            return this;
        }
    }
}
