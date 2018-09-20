package io.improbable.keanu.plating;

import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexDictionary;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.VertexLabelException;

/**
 * PlateBuilder allows plates to be constructed in steps
 *
 * @param <T> The data type provided to user-provided plate
 *            factory function, if building from data
 */
public class PlateBuilder<T> {

    private static final String PROXY_LABEL_MARKER = "proxy_for";
    private VertexDictionary initialState;
    private Map<VertexLabel, VertexLabel> transitionMapping = Collections.emptyMap();

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
         * @throws VertexLabelException which can occur e.g. if the labels don't marry up in the transition mapping
         */
        Plates build();
    }

    public static VertexLabel proxyFor(VertexLabel label) {
        return label.withExtraNamespace(PROXY_LABEL_MARKER);
    }


    public PlateBuilder<T> withInitialState(Vertex<?> initialState) {
        return withInitialState(VertexDictionary.of(initialState));
    }

    public PlateBuilder<T> withInitialState(VertexDictionary initialState) {
        this.initialState = initialState;
        return this;
    }

    public PlateBuilder<T> withTransitionMapping(Map<VertexLabel, VertexLabel> transitionMapping) {
        this.transitionMapping = transitionMapping;
        return this;
    }

    /**
     * Build a fixed number of plates without additional data
     *
     * @param count count
     * @return A builder with count set
     */
    public FromCount count(int count) {
        return new FromCount(count, initialState);
    }

    /**
     * Build an unspecified number of plates with data from an iterator
     *
     * @param iterator iterator
     * @return A builder with data set
     */
    public FromIterator fromIterator(Iterator<T> iterator) {
        return new FromIterator(iterator, 0, initialState, transitionMapping);
    }

    /**
     * Build a number of plates with data from an iterator
     *
     * @param iterator iterator
     * @param sizeHint A hint of the iterator cardinality. Does not need to be exact
     * @return A builder with data set
     */
    public FromIterator fromIterator(Iterator<T> iterator, int sizeHint) {
        return new FromIterator(iterator, sizeHint, initialState, transitionMapping);
    }

    /**
     * An intermediate builder, with a set count
     */
    public class FromCount implements PlateCount {
        private final int count;
        private final VertexDictionary initialState;

        public FromCount(int count, VertexDictionary initialState) {
            this.count = count;
            this.initialState = initialState;
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
            return new FromCountFactory(factory, this, initialState, transitionMapping);
        }
    }

    /**
     * An intermediate builder, with a set data iterator
     */
    public class FromIterator implements PlateData<T> {
        private Iterator<T> iterator;
        private int size;
        private final VertexDictionary initialState;

        private FromIterator(Iterator<T> iterator, int size, VertexDictionary initialState, Map<VertexLabel, VertexLabel> transitionMapping) {
            this.iterator = iterator;
            this.size = size;
            this.initialState = initialState;
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
            return new FromDataFactory(factory, this, size, initialState);
        }
    }

    /**
     * Build Plates from some provided Data
     */
    public class FromDataFactory implements PlateFactory {
        private BiConsumer<Plate, T> factory;
        private PlateData<T> data;
        private int size;
        private final VertexDictionary initialState;

        private FromDataFactory(BiConsumer<Plate, T> factory, PlateData<T> data, int size, VertexDictionary initialState) {
            this.factory = factory;
            this.data = data;
            this.size = size;
            this.initialState = initialState;
        }

        public Plates build() throws VertexLabelException {
            Plates plates = new Plates(this.size);
            Iterator<T> iter = data.getIterator();
            VertexDictionary previousVertices = initialState;
            while (iter.hasNext()) {
                Plate plate = new Plate();
                factory.accept(plate, iter.next());
                connectTransitionVariables(previousVertices, plate, transitionMapping);
                plates.add(plate);
                previousVertices = plate;
            }
            return plates;
        }
    }

    private void connectTransitionVariables(VertexDictionary candidateVertices, Plate plate, Map<VertexLabel, VertexLabel> transitionMapping) throws VertexLabelException {
        Collection<Vertex<?>> proxyVertices = plate.getProxyVertices();
        if (candidateVertices == null && !proxyVertices.isEmpty()) {
            throw new IllegalArgumentException("You must provide a base case for the Transition Vertices - use withInitialState()");
        }
        for (Vertex<?> proxy : proxyVertices) {
            VertexLabel proxyLabel = proxy.getLabel().withoutOuterNamespace();
            VertexLabel defaultParentLabel = getDefaultParentLabel(proxyLabel);
            VertexLabel parentLabel = transitionMapping.getOrDefault(proxyLabel, defaultParentLabel);

            if (parentLabel == null) {
                throw new VertexLabelException("Cannot find transition mapping for " + proxy.getLabel());
            }

            Vertex<?> parent = candidateVertices.get(parentLabel);
            if (parent == null) {
                throw new VertexLabelException("Cannot find VertexLabel " + parentLabel);
            }
            proxy.setParents(parent);
        }
    }

    private VertexLabel getDefaultParentLabel(VertexLabel proxyLabel) {
        String outerNamespace = proxyLabel.getOuterNamespace().orElse(null);
        if (PROXY_LABEL_MARKER.equals(outerNamespace)) {
            return proxyLabel.withoutOuterNamespace();
        } else {
            return null;
        }
    }

    /**
     * Build some number of plates
     */
    public class FromCountFactory implements PlateFactory {
        private Consumer<Plate> factory;
        private PlateCount count;

        private FromCountFactory(Consumer<Plate> factory, PlateCount count, VertexDictionary initialState, Map<VertexLabel, VertexLabel> transitionMapping) {
            this.factory = factory;
            this.count = count;
        }


        public Plates build() throws VertexLabelException {
            Plates plates = new Plates(count.getCount());
            VertexDictionary previousPlate = initialState;
            for (int i = 0; i < count.getCount(); i++) {
                Plate plate = new Plate();
                factory.accept(plate);
                connectTransitionVariables(previousPlate, plate, transitionMapping);
                plates.add(plate);
                previousPlate = plate;
            }
            return plates;
        }
    }
}
