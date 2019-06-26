package io.improbable.keanu.tensor.generic;

import io.improbable.keanu.tensor.buffer.JVMBuffer;
import io.improbable.keanu.tensor.dbl.DoubleBuffer;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

public class GenericBuffer {

    public static final class GenericArrayWrapperFactory<T> implements JVMBuffer.ArrayWrapperFactory<T, PrimitiveGenericWrapper<T>> {

        @Override
        public final GenericBuffer.PrimitiveGenericWrapper<T> createNew(final int size) {
            if (size == 1) {
                return new GenericBuffer.GenericWrapper<>(null);
            } else {
                return new GenericBuffer.GenericArrayWrapper<>((T[]) (new Object[size]));
            }
        }

        public final GenericBuffer.PrimitiveGenericWrapper<T> create(T[] data) {
            if (data.length == 1) {
                return new GenericBuffer.GenericWrapper<>(data[0]);
            } else {
                return new GenericBuffer.GenericArrayWrapper<>(data);
            }
        }
    }

    public interface PrimitiveGenericWrapper<T> extends JVMBuffer.PrimitiveArrayWrapper<T, PrimitiveGenericWrapper<T>> {
        T[] asArray();
    }

    public static final class GenericArrayWrapper<T> implements PrimitiveGenericWrapper<T> {

        private final T[] array;

        public GenericArrayWrapper(final T[] array) {
            this.array = array;
        }

        @Override
        public T get(final int index) {
            return array[index];
        }

        @Override
        public void set(final T value, final int index) {
            array[index] = value;
        }

        @Override
        public int getLength() {
            return array.length;
        }

        @Override
        public PrimitiveGenericWrapper<T> copy() {
            return new GenericArrayWrapper<>(Arrays.copyOf(array, array.length));
        }

        @Override
        public void copyFrom(JVMBuffer.PrimitiveArrayWrapper<T, ?> src, int srcPos, int destPos, int length) {
            if (src instanceof GenericArrayWrapper) {
                System.arraycopy(((GenericArrayWrapper) src).array, srcPos, array, destPos, length);
            } else {
                for (int i = 0; i < length; i++) {
                    array[destPos + i] = src.get(srcPos + i);
                }
            }
        }

        @Override
        public T[] asArray() {
            return array;
        }

        @Override
        public void applyRight(BiFunction<T, T, T> mapper, T rightArg) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(array[i], rightArg);
            }
        }

        @Override
        public void applyLeft(BiFunction<T, T, T> mapper, T leftArg) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(leftArg, array[i]);
            }
        }

        @Override
        public void apply(Function<T, T> mapper) {
            for (int i = 0; i < array.length; i++) {
                array[i] = mapper.apply(array[i]);
            }
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
        public PrimitiveGenericWrapper<T> copy() {
            return new GenericWrapper<>(value);
        }

        @Override
        public T[] asArray() {
            return (T[]) (new Object[]{value});
        }

    }
}
