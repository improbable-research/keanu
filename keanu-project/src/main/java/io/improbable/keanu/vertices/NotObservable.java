package io.improbable.keanu.vertices;

import java.util.Optional;

public class NotObservable<T> implements Observable<T> {
    // package private - because it's created by the factory method Observable.observableTypeFor
    NotObservable() {
    }

    @Override
    public void observe(T value) {
        throw new UnsupportedOperationException("This type of vertex does not support being observed");
    }

    @Override
    public void unobserve() {
        // Do nothing:
        // This is allowed, since it's common for algorithms to unobserve all vertices
        // e.g. in io.improbable.keanu.network.NetworkSnapshot.apply
    }

    @Override
    public Optional<T> getObservedValue() {
        return Optional.empty();
    }

    @Override
    public boolean isObserved() {
        return false;
    }
}
