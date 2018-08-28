package io.improbable.keanu.vertices.generic;

import io.improbable.keanu.vertices.Vertex;

/**
 * A Proxy vertex is one which has no parents at creation time but can at a later point "hook up" a parent node.  All
 * Vertex operations are then passed through to the parent (as if the they'd been called on the parent directly)
 */
public interface ProxyVertex<T extends Vertex<?>> {
    public void setParent(T newParent);

    public boolean hasParent();
}
