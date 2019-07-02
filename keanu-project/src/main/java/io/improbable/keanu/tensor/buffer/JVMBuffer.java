package io.improbable.keanu.tensor.buffer;

import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;

public class JVMBuffer {

    public interface PrimitiveArrayWrapper<T, IMPL extends PrimitiveArrayWrapper<T, IMPL>> {

        T[] asArray();

        T get(int index);

        IMPL set(T value, int index);

        int getLength();

        IMPL copy();

        IMPL copyFrom(PrimitiveArrayWrapper<T, ?> src, int srcPos, int destPos, int length);

        IMPL apply(Function<T, T> mapper);

        IMPL applyRight(BiFunction<T, T, T> mapper, T rightArg);

        IMPL applyLeft(BiFunction<T, T, T> mapper, T leftArg);
    }

    public static abstract class SingleValueWrapper<T, IMPL extends PrimitiveArrayWrapper<T, IMPL>> implements PrimitiveArrayWrapper<T, IMPL> {

        protected T value;

        public SingleValueWrapper(final T value) {
            this.value = value;
        }

        @Override
        public T get(final int index) {
            return value;
        }

        @Override
        public IMPL set(final T value, final int index) {
            this.value = value;
            return getThis();
        }

        @Override
        public int getLength() {
            return 1;
        }

        public IMPL copyFrom(JVMBuffer.PrimitiveArrayWrapper<T, ?> src, int srcPos, int destPos, int length) {
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
        T createNew(int size);

        T createNew(D value);
    }

    public interface PrimitiveNumberWrapperFactory<D, T extends PrimitiveArrayWrapper<D, T>> extends ArrayWrapperFactory<D, T> {
        T zeroes(int size);

        T ones(int size);
    }

}
