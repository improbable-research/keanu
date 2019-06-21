package io.improbable.keanu.tensor.buffer;

import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;

public class JVMBuffer {

    public interface PrimitiveArrayWrapper<T> {
        T get(int index);

        void set(T value, int index);

        int getLength();

        PrimitiveArrayWrapper<T> copy();

        void apply(Function<T, T> mapper);

        void applyRight(BiFunction<T, T, T> mapper, T rightArg);

        void applyLeft(BiFunction<T, T, T> mapper, T leftArg);
    }

    public static abstract class SingleValueWrapper<T> implements PrimitiveArrayWrapper<T> {

        T value;

        public SingleValueWrapper(final T value) {
            this.value = value;
        }

        @Override
        public T get(final int index) {
            return value;
        }

        @Override
        public void set(final T value, final int index) {
            this.value = value;
        }

        @Override
        public int getLength() {
            return 1;
        }

        public abstract PrimitiveArrayWrapper<T> copy();

        @Override
        public void apply(Function<T, T> mapper) {
            value = mapper.apply(value);
        }

        @Override
        public void applyRight(BiFunction<T, T, T> mapper, T rightArg) {
            value = mapper.apply(value, rightArg);
        }

        @Override
        public void applyLeft(BiFunction<T, T, T> mapper, T leftArg) {
            value = mapper.apply(leftArg, value);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            SingleValueWrapper<?> that = (SingleValueWrapper<?>) o;
            return value.equals(that.value);
        }

        @Override
        public int hashCode() {
            return Objects.hash(value);
        }
    }

    public interface ArrayWrapperFactory<D, T extends PrimitiveArrayWrapper<D>> {
        T createNew(int size);
    }

}
