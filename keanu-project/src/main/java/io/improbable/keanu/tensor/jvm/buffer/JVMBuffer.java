package io.improbable.keanu.tensor.jvm.buffer;

import io.improbable.keanu.tensor.bool.BooleanBuffer;

import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;

public class JVMBuffer {

    public interface PrimitiveArrayWrapper<T, IMPL extends PrimitiveArrayWrapper<T, IMPL>> {

        Object[] asArray();

        T get(long index);

        IMPL set(T value, long index);

        long getLength();

        IMPL copy();

        IMPL copyFrom(PrimitiveArrayWrapper<T, ?> src, long srcPos, long destPos, long length);

        IMPL apply(Function<T, T> mapper);

        IMPL applyRight(BiFunction<T, T, T> mapper, T rightArg);

        IMPL applyLeft(BiFunction<T, T, T> mapper, T leftArg);

        default T reduce(T initial, BiFunction<T, T, T> reducer) {
            T result = initial;
            for (int i = 0; i < getLength(); i++) {
                result = reducer.apply(result, get(i));
            }
            return result;
        }

        BooleanBuffer.PrimitiveBooleanWrapper equal(T that);
    }

    public static abstract class SingleValueWrapper<T, IMPL extends PrimitiveArrayWrapper<T, IMPL>> implements PrimitiveArrayWrapper<T, IMPL> {

        protected T value;

        public SingleValueWrapper(final T value) {
            this.value = value;
        }

        @Override
        public T get(final long index) {
            return value;
        }

        @Override
        public IMPL set(final T value, final long index) {
            this.value = value;
            return getThis();
        }

        @Override
        public long getLength() {
            return 1;
        }

        public IMPL copyFrom(JVMBuffer.PrimitiveArrayWrapper<T, ?> src, long srcPos, long destPos, long length) {
            if (length == 1 && destPos == 0) {
                value = src.get(srcPos);
            } else if (length > 1 || length < 0 || destPos != 0) {
                throw new IndexOutOfBoundsException();
            }
            return getThis();
        }

        @Override
        public IMPL apply(Function<T, T> mapper) {
            value = mapper.apply(value);
            return getThis();
        }

        @Override
        public IMPL applyRight(BiFunction<T, T, T> mapper, T rightArg) {
            value = mapper.apply(value, rightArg);
            return getThis();
        }

        @Override
        public IMPL applyLeft(BiFunction<T, T, T> mapper, T leftArg) {
            value = mapper.apply(leftArg, value);
            return getThis();
        }

        @Override
        public T reduce(T initial, BiFunction<T, T, T> reducer) {
            return reducer.apply(initial, value);
        }

        protected abstract IMPL getThis();

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            SingleValueWrapper<?, ?> that = (SingleValueWrapper<?, ?>) o;
            return value.equals(that.value);
        }

        @Override
        public int hashCode() {
            return Objects.hash(value);
        }
    }

    public interface ArrayWrapperFactory<D, T extends PrimitiveArrayWrapper<D, T>> {
        T createNew(long size);

        T createNew(D value);
    }

    public interface PrimitiveNumberWrapperFactory<D, T extends PrimitiveArrayWrapper<D, T>> extends ArrayWrapperFactory<D, T> {
        T zeroes(long size);

        T ones(long size);
    }

}
