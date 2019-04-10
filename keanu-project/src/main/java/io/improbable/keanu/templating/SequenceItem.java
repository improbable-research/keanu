package io.improbable.keanu.templating;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.vertices.ProxyVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexDictionary;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.nonprobabilistic.BooleanProxyVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleProxyVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.IntegerProxyVertex;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static com.google.common.collect.ImmutableMap.copyOf;
import static java.lang.Integer.parseInt;

public class SequenceItem implements VertexDictionary {

    private static final String NAME_PREFIX = "Sequence_Item_";
    private static Pattern NAME_REGEX = Pattern.compile(NAME_PREFIX + "-?[\\d]+$");

    private Map<VertexLabel, Vertex<?>> contents;
    private int index;
    private int uniqueSequenceIdentifier;
    private String sequenceName;

    public SequenceItem(int index, int uniqueSequenceIdentifier) {
        this(index, uniqueSequenceIdentifier, null);
    }

    public SequenceItem(int index, int uniqueSequenceIdentifier, String sequenceName) {
        this.contents = new HashMap<>();
        this.index = index;
        this.uniqueSequenceIdentifier = uniqueSequenceIdentifier;
        this.sequenceName = sequenceName;
    }

    public <T extends Vertex<?>> void addAll(T... vertices) {
        addAll(ImmutableList.copyOf(vertices));
    }

    public <T extends Vertex<?>> void addAll(Collection<T> vertices) {
        vertices.forEach(v -> add(v));
    }

    public <T extends Vertex<?>> void addAll(Map<VertexLabel, T> vertices) {
        vertices.entrySet().forEach(v -> add(v.getKey(), v.getValue()));
    }

    public <T extends Vertex<?>> T add(T v) {
        return add(v.getLabel(), v);
    }

    public <T extends Vertex<?>> T add(VertexLabel label, T v) {
        if (label == null) {
            throw new SequenceConstructionException("Vertex " + v + " must contain a label in order to be added to a sequence item");
        }
        String outerNamespace = label.getOuterNamespace().orElse("");
        if (NAME_REGEX.matcher(outerNamespace).matches()) {
            throw new SequenceConstructionException("Vertex " + v + " has already been added to " + outerNamespace);
        }
        label = scoped(label);
        if (contents.containsKey(label)) {
            throw new IllegalArgumentException("Key " + label + " already exists");
        }
        contents.put(label, v);
        v.setLabel(label);
        return v;
    }

    /**
     * @return Returns a map of vertex labels to vertices, which covers all of the vertices that have been explicitly
     * added to this sequence item.
     */
    public Map<VertexLabel, Vertex<?>> getContents() {
        return copyOf(this.contents);
    }

    /**
     * @return the index of the sequence item in the overall {@link Sequence}
     */
    public int getIndex() {
        return this.index;
    }

    private String getName() {
        return NAME_PREFIX + this.index;
    }

    private VertexLabel scoped(VertexLabel label) {
        VertexLabel scopedLabel = label
            .withExtraNamespace(String.valueOf(this.uniqueSequenceIdentifier))
            .withExtraNamespace(getName());

        if (this.sequenceName != null) {
            scopedLabel = scopedLabel.withExtraNamespace(this.sequenceName);
        }
        return scopedLabel;
    }

    @Override
    public <V extends Vertex<?>> V get(VertexLabel label) {
        Vertex<?> vertex = contents.getOrDefault(label, contents.get(scoped(label)));

        if (vertex == null) {
            throw new IllegalArgumentException("Cannot find VertexLabel " + label);
        }
        return (V) vertex;
    }

    @Override
    public SequenceItem withExtraEntries(Map<VertexLabel, Vertex<?>> extraEntries) {
        SequenceItem item = new SequenceItem(this.index, this.uniqueSequenceIdentifier, this.sequenceName);
        item.addAll(contents);
        item.addAll(extraEntries);
        return item;
    }

    public Collection<Vertex<?>> getProxyVertices() {
        return contents.values().stream()
            .filter(v -> v instanceof ProxyVertex)
            .collect(Collectors.toList());
    }

    /**
     * This method creates a {@link ProxyVertex} and adds it to the sequence item
     * It also adds this vertex to the sequence item to represent the vertex with this label in the previous sequence item.
     * @param label the label of the corresponding vertex from the previous item.
     * @return a newly created {@link ProxyVertex}
     */
    public DoubleProxyVertex addDoubleProxyFor(VertexLabel label) {
        return addProxyFor(label, DoubleProxyVertex::new);
    }

    /**
     * This method creates a {@link ProxyVertex} and adds it to the sequence item
     * It also adds this vertex to the sequence item to represent the vertex with this label in the previous sequence item.
     * @param label the label of the corresponding vertex from the previous item.
     * @param shape the shape of the corresponding vertex from the previous item.
     * @return a newly created {@link ProxyVertex}
     */
    public DoubleProxyVertex addDoubleProxyFor(VertexLabel label, long[] shape) {
        return addProxyFor(label, shape, DoubleProxyVertex::new);
    }

    /**
     * This method creates a {@link ProxyVertex} and adds it to the sequence item
     * It also adds this vertex to the sequence item to represent the vertex with this label in the previous sequence item.
     * @param label the label of the corresponding vertex from the previous item.
     * @return a newly created {@link ProxyVertex}
     */
    public IntegerProxyVertex addIntegerProxyFor(VertexLabel label) {
        return addProxyFor(label, IntegerProxyVertex::new);
    }

    /**
     * This method creates a {@link ProxyVertex} and adds it to the sequence item
     * It also adds this vertex to the sequence item to represent the vertex with this label in the previous sequence item.
     * @param label the label of the corresponding vertex from the previous item.
     * @param shape the shape of the corresponding vertex from the previous item.
     * @return a newly created {@link ProxyVertex}
     */
    public IntegerProxyVertex addIntegerProxyFor(VertexLabel label, long[] shape) {
        return addProxyFor(label, shape, IntegerProxyVertex::new);
    }

    /**
     * This method creates a {@link ProxyVertex} and adds it to the sequence item
     * It also adds this vertex to the sequence item to represent the vertex with this label in the previous sequence item.
     * @param label the label of the corresponding vertex from the previous item.
     * @return a newly created {@link ProxyVertex}
     */
    public BooleanProxyVertex addBooleanProxyFor(VertexLabel label) {
        return addProxyFor(label, BooleanProxyVertex::new);
    }

    /**
     * This method creates a {@link ProxyVertex} and adds it to the sequence item
     * It also adds this vertex to the sequence item to represent the vertex with this label in the previous sequence item.
     * @param label the label of the corresponding vertex from the previous item.
     * @param shape the shape of the corresponding vertex from the previous item.
     * @return a newly created {@link ProxyVertex}
     */
    public BooleanProxyVertex addBooleanProxyFor(VertexLabel label, long[] shape) {
        return addProxyFor(label, shape, BooleanProxyVertex::new);
    }

    private <T extends Vertex<?>> T addProxyFor(VertexLabel label, Function<VertexLabel, T> factoryMethod) {
        return addProxyFor(label, null, (shape, vertexLabel) -> factoryMethod.apply(vertexLabel));
    }

    private <T extends Vertex<?>> T addProxyFor(VertexLabel label, long[] shape, BiFunction<long[], VertexLabel, T> factoryMethod) {
        VertexLabel proxyLabel = SequenceBuilder.proxyLabelFor(label);
        T newVertex = factoryMethod.apply(shape, proxyLabel);
        this.add(newVertex);
        return newVertex;
    }

    /**
     * Tries to retrieve a SequenceItem index from a vertex label of a vertex in a sequence
     * @param label label of the vertex being parsed.
     * @return empty if the vertex is not in a SequenceItem. Otherwise returns index of the vertex in the sequence item.
     */
    static Optional<Integer> getSequenceItemIndex(VertexLabel label) {
        String outerNamespace = label.getOuterNamespace().orElse(null);
        if (outerNamespace == null) {
            return Optional.empty();
        }
        if (outerNamespace.startsWith(SequenceItem.NAME_PREFIX)) {
            return Optional.of(parseInt(outerNamespace.replaceFirst(SequenceItem.NAME_PREFIX, "")));
        }
        outerNamespace = label.withoutOuterNamespace().getOuterNamespace().orElse(null);
        if (outerNamespace == null) {
            return Optional.empty();
        }
        return Optional.of(parseInt(outerNamespace.replaceFirst(SequenceItem.NAME_PREFIX, "")));
    }


    /**
     * Tries to get the unique sequence name from a vertex label
     * @param label the label of a vertex in a sequence
     * @return will return empty if the label cannot be parsed as a label of a vertex from a sequence with a name.
     */
    static Optional<String> getSequenceName(VertexLabel label) {
        Optional<String> outerNamespace = label.getOuterNamespace();
        Optional<String> penultimateOuterNamespace = label.withoutOuterNamespace().getOuterNamespace();
        if (penultimateOuterNamespace.isPresent() && penultimateOuterNamespace.get().startsWith(NAME_PREFIX)) {
            return outerNamespace;
        }
        return Optional.empty();
    }
    /**
     * Tries to get the unique sequence hash from a vertex label
     * @param label the label of a vertex in a sequence
     * @return will return empty if the label cannot be parsed as a label of a vertex from a sequence
     */
    static int getSequenceHash(VertexLabel label, boolean sequenceHasName) {
        VertexLabel withHashAsOuterNamespace = label.withoutOuterNamespace();
        if (sequenceHasName) {
            withHashAsOuterNamespace = withHashAsOuterNamespace.withoutOuterNamespace();
        }
        String hashLabel = withHashAsOuterNamespace.getOuterNamespace()
            .orElseThrow(() -> new SequenceConstructionException("Could not parse the sequence hash in the vertex label"));
        return parseInt(hashLabel);
    }
}
