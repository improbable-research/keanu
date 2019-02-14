package io.improbable.keanu.plating;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexDictionary;
import io.improbable.keanu.vertices.VertexLabel;

import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * SequenceBuilder allows sequences to be constructed in steps
 *
 * @param <T> The data type provided to user-provided sequence
 *            factory function, if building from data
 */
public class SequenceBuilder<T> {

    private static final String PROXY_LABEL_MARKER = "proxy_for";
    private VertexDictionary initialState;
    private Map<VertexLabel, VertexLabel> transitionMapping = Collections.emptyMap();

    private interface ItemCount {
        int getCount();
    }

    private interface SequenceData<T> {
        Iterator<T> getIterator();
    }

    private interface SequenceFactory {
        /**
         * Build sequence from current factory settings
         *
         * @return Sequence
         * @throws SequenceConstructionException which can occur e.g. if the labels don't marry up in the transition mapping
         */
        Sequence build();
    }

    public static VertexLabel proxyFor(VertexLabel label) {
        return label.withExtraNamespace(PROXY_LABEL_MARKER);
    }


    public SequenceBuilder<T> withInitialState(Vertex<?> vertex) {
        return withInitialState(VertexDictionary.of(vertex));
    }

    public SequenceBuilder<T> withInitialState(VertexLabel label, Vertex<?> vertex) {
        return withInitialState(VertexDictionary.backedBy(ImmutableMap.of(label, vertex)));
    }

    public SequenceBuilder<T> withInitialState(VertexDictionary initialState) {
        this.initialState = initialState;
        return this;
    }

    public SequenceBuilder<T> withTransitionMapping(Map<VertexLabel, VertexLabel> transitionMapping) {
        this.transitionMapping = transitionMapping;
        return this;
    }

    /**
     * Build a fixed number of sequence items without additional data
     *
     * @param count count
     * @return A builder with count set
     */
    public FromCount count(int count) {
        return new FromCount(count, initialState);
    }

    /**
     * Build an unspecified number of sequence items with data from an iterator
     *
     * @param iterator iterator
     * @return A builder with data set
     */
    public FromIterator fromIterator(Iterator<T> iterator) {
        return new FromIterator(iterator, 0, initialState, transitionMapping);
    }

    /**
     * Build a number of sequence items with data from an iterator
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
    public class FromCount implements ItemCount {
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
         * Set the SequenceItem factory method, taking no additional data
         *
         * @param factory a sequence factory
         * @return A builder with count and sequence factory set
         */
        public FromCountFactory withFactory(Consumer<SequenceItem> factory) {
            return new FromCountFactory(factory, this, initialState, transitionMapping);
        }
    }

    /**
     * An intermediate builder, with a set data iterator
     */
    public class FromIterator implements SequenceData<T> {
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
         * Set the SequenceItem factory method, taking additional data
         *
         * @param factory a sequence factory
         * @return A builder with data and sequence factory set
         */
        public FromDataFactory withFactory(BiConsumer<SequenceItem, T> factory) {
            return new FromDataFactory(factory, this, size, initialState);
        }
    }

    /**
     * Build Sequence from some provided Data
     */
    public class FromDataFactory implements SequenceFactory {
        private BiConsumer<SequenceItem, T> factory;
        private SequenceData<T> data;
        private int size;
        private final VertexDictionary initialState;

        private FromDataFactory(BiConsumer<SequenceItem, T> factory, SequenceData<T> data, int size, VertexDictionary initialState) {
            this.factory = factory;
            this.data = data;
            this.size = size;
            this.initialState = initialState;
        }

        public Sequence build() throws SequenceConstructionException {
            Sequence sequence = new Sequence(this.size);
            Iterator<T> iter = data.getIterator();
            VertexDictionary previousVertices = initialState;
            while (iter.hasNext()) {
                SequenceItem item = new SequenceItem();
                factory.accept(item, iter.next());
                connectTransitionVariables(previousVertices, item, transitionMapping);
                sequence.add(item);
                previousVertices = item;
            }
            return sequence;
        }
    }

    private void connectTransitionVariables(VertexDictionary candidateVertices, SequenceItem item, Map<VertexLabel, VertexLabel> transitionMapping) throws SequenceConstructionException {
        Collection<Vertex<?>> proxyVertices = item.getProxyVertices();

        for (Vertex<?> proxy : proxyVertices) {
            VertexLabel proxyLabel = proxy.getLabel().withoutOuterNamespace();
            VertexLabel defaultParentLabel = getDefaultParentLabel(proxyLabel);
            VertexLabel parentLabel = transitionMapping.getOrDefault(proxyLabel, defaultParentLabel);

            if (parentLabel == null) {
                throw new SequenceConstructionException("Cannot find transition mapping for " + proxy.getLabel());
            }

            if (candidateVertices == null) {
                throw new IllegalArgumentException("You must provide a base case for the Transition Vertices - use withInitialState()");
            }

            Vertex<?> parent = candidateVertices.get(parentLabel);
            if (parent == null) {
                throw new SequenceConstructionException("Cannot find VertexLabel " + parentLabel);
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
     * Build some number of sequence items
     */
    public class FromCountFactory implements SequenceFactory {
        private Consumer<SequenceItem> factory;
        private ItemCount count;

        private FromCountFactory(Consumer<SequenceItem> factory, ItemCount count, VertexDictionary initialState, Map<VertexLabel, VertexLabel> transitionMapping) {
            this.factory = factory;
            this.count = count;
        }


        public Sequence build() throws SequenceConstructionException {
            Sequence sequence = new Sequence(count.getCount());
            VertexDictionary previousItem = initialState;
            for (int i = 0; i < count.getCount(); i++) {
                SequenceItem item = new SequenceItem();
                factory.accept(item);
                connectTransitionVariables(previousItem, item, transitionMapping);
                sequence.add(item);
                previousItem = item;
            }
            return sequence;
        }
    }
}
