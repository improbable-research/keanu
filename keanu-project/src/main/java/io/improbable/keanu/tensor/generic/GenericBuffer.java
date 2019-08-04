package io.improbable.keanu.tensor.generic;

import com.google.common.primitives.Ints;
import io.improbable.keanu.tensor.jvm.buffer.JVMBuffer;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

public class GenericBuffer {

    public static final class GenericArrayWrapperFactory<T> implements JVMBuffer.ArrayWrapperFactory<T, PrimitiveGenericWrapper<T>> {

        @Override
        public final GenericBuffer.PrimitiveGenericWrapper<T> createNew(final long size) {
            if (size == 1) {
                return new GenericBuffer.GenericWrapper<>(null);
            } else {
                return new GenericBuffer.GenericArrayWrapper<>((new Object[Ints.checkedCast(size)]));
            }
        }

        @Override
        public PrimitiveGenericWrapper<T> createNew(T value) {
            return new GenericBuffer.GenericWrapper<>(value);
        }

        public final GenericBuffer.PrimitiveGenericWrapper<T> create(Object[] data) {
            if (data.length == 1) {
                return new GenericBuffer.GenericWrapper<>((T) data[0]);
            } else {
                return new GenericBuffer.GenericArrayWrapper<>(data);
            }
        }
    }

    public interface PrimitiveGenericWrapper<T> extends JVMBuffer.PrimitiveArrayWrapper<T, PrimitiveGenericWrapper<T>> {
    }

    public static final class GenericArrayWrapper<T> implements PrimitiveGenericWrapper<T> {

        private final Object[] array;

        public GenericArrayWrapper(final Object[] array) {
            this.array = array;
        }

        @Override
        public T get(final long index) {
            return (T) array[Ints.checkedCast(index)];
        }

        @Override
        public GenericArrayWrapper<T> set(final T value, final long index) {
            array[Ints.checkedCast(index)] = value;
            return this;
        }

        @Override
        public long getLength() {
            return array.length;
        }

        @Override
        public PrimitiveGenericWrapper<T> copy() {
            return new GenericArrayWrapper<>(Arrays.copyOf(array, array.length));
        }

        @Override
        public GenericArrayWrapper<T> copyFrom(JVMBuffer.PrimitiveArrayWrapper<T, ?> src, long srcPos, long destPos, long length) {
            if (src instanceof GenericArrayWrapper) {
                System.arraycopy(((GenericArrayWrapper) src).array, Ints.checkedCast(srcPos), array, Ints.checkedCast(destPos), Ints.checkedCast(length));
            } else {
                for (int i = 0; i < length; i++) {
                    array[Ints.checkedCast(destPos + i)] = src.get(srcPos + i);
                }
            }
            return this;
        }

        @Override
        public Object[] asArray() {
            return array;
        }

        @Override
        public GenericArrayWrapper<T> applyRight(BiFunction<T, T, T> mapper, T rightArg) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply((T) array[i], rightArg);
            }
            return this;
        }

        @Override
        public GenericArrayWrapper<T> applyLeft(BiFunction<T, T, T> mapper, T leftArg) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(leftArg, (T) array[i]);
            }
            return this;
        }

        @Override
        public GenericArrayWrapper<T> apply(Function<T, T> mapper) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply((T) array[i]);
            }
            return this;
        }

        public boolean equals(final Object o) {
            if (o == this) return true;
            if (!(o instanceof PrimitiveGenericWrapper)) return false;
            final PrimitiveGenericWrapper other = (PrimitiveGenericWrapper) o;
            if (!Arrays.equals(this.array, other.asArray())) return false;
            return true;
        }

        public int hashCode() {
            final int PRIME = 59;
            int result = 1;
            result = result * PRIME + Arrays.hashCode(this.array);
            return result;
        }

    }

    public static final class GenericWrapper<T> extends JVMBuffer.SingleValueWrapper<T, PrimitiveGenericWrapper<T>> implements PrimitiveGenericWrapper<T> {

        public GenericWrapper(final T value) {
            super(value);
        }

        @Override
        protected PrimitiveGenericWrapper<T> getThis() {
            return this;
        }

        @Override
        public PrimitiveGenericWrapper<T> copy() {
            return new GenericWrapper<>(value);
        }

        @Override
        public Object[] asArray() {
            return new Object[]{value};
        }

    }
}
