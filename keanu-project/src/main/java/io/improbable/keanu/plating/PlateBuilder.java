package io.improbable.keanu.plating;

import java.util.Iterator;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * PlateBuilder allows plates to constructed in steps
 *
 * @param <T> The data type provided to user-provided plate
 *            factory function, if building from data
 */
public class PlateBuilder<T> {
    private interface PlateCount {
        int getCount();
    }

    private interface PlateData<T> {
        Iterator<T> getIterator();
    }

    private interface PlateFactory {
        /**
         * Build plates from current factory settings
         *
         * @return Collection of all created plates
         */
        Plates build();
    }

    /**
     * Build a fixed number of plates without additional data
     *
     * @param count count
     * @return A builder with count set
     */
    public FromCount count(int count) {
        return new FromCount(count);
    }

    /**
     * Build an unspecified number of plates with data from an iterator
     *
     * @param iterator iterator
     * @return A builder with data set
     */
    public FromIterator fromIterator(Iterator<T> iterator) {
        return new FromIterator(iterator, 0);
    }

    /**
     * Build a number of plates with data from an iterator
     *
     * @param iterator iterator
     * @param sizeHint A hint of the iterator cardinality. Does not need to be exact
     * @return A builder with data set
     */
    public FromIterator fromIterator(Iterator<T> iterator, int sizeHint) {
        return new FromIterator(iterator, sizeHint);
    }

    /**
     * An intermediate builder, with a set count
     */
    public class FromCount implements PlateCount {
        private int count;

        public FromCount(int count) {
            this.count = count;
        }

        public int getCount() {
            return this.count;
        }

        /**
         * Set the Plate factory method, taking no additional data
         *
         * @param factory a plate factory
         * @return A builder with count and plate factory set
         */
        public FromCountFactory withFactory(Consumer<Plate> factory) {
            return new FromCountFactory(factory, this);
        }
    }

    /**
     * An intermediate builder, with a set data iterator
     */
    public class FromIterator implements PlateData<T> {
        private Iterator<T> iterator;
        private int size;

        private FromIterator(Iterator<T> iterator, int size) {
            this.iterator = iterator;
            this.size = size;
        }

        public Iterator<T> getIterator() {
            return this.iterator;
        }

        /**
         * Set the Plate factory method, taking additional data
         *
         * @param factory a plate factory
         * @return A builder with data and plate factory set
         */
        public FromDataFactory withFactory(BiConsumer<Plate, T> factory) {
            return new FromDataFactory(factory, this, size);
        }
    }

    /**
     * Build Plates from some provided Data
     */
    public class FromDataFactory implements PlateFactory {
        private BiConsumer<Plate, T> factory;
        private PlateData<T> data;
        private int size;

        private FromDataFactory(BiConsumer<Plate, T> factory, PlateData<T> data, int size) {
            this.factory = factory;
            this.data = data;
            this.size = size;
        }

        public Plates build() {
            Plates plates = new Plates(this.size);
            Iterator<T> iter = data.getIterator();
            while (iter.hasNext()) {
                Plate plate = new Plate();
                factory.accept(plate, iter.next());
                plates.add(plate);
            }
            return plates;
        }
    }

    /**
     * Build some number of plates
     */
    public class FromCountFactory implements PlateFactory {
        private Consumer<Plate> factory;
        private PlateCount count;

        private FromCountFactory(Consumer<Plate> factory, PlateCount count) {
            this.factory = factory;
            this.count = count;
        }


        public Plates build() {
            Plates plates = new Plates(count.getCount());
            for (int i = 0; i < count.getCount(); i++) {
                Plate plate = new Plate();
                factory.accept(plate);
                plates.add(plate);
            }
            return plates;
        }
    }
}
