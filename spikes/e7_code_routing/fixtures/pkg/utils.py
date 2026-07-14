"""Small grab-bag of helpers, used as a "navigate" / "edit" fixture target."""


def slugify(text: str) -> str:
    """Lowercase, strip, replace runs of whitespace with a single hyphen."""
    return "-".join(text.strip().lower().split())


class Cache:
    """A tiny bounded dict cache, oldest key evicted first."""

    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self._data: dict = {}

    def put(self, key, value) -> None:
        """Insert key/value, evicting the oldest entry if over capacity."""
        if len(self._data) >= self.capacity and key not in self._data:
            oldest = next(iter(self._data))
            del self._data[oldest]
        self._data[key] = value

    def get(self, key, default=None):
        """Look up a key, returning default when absent."""
        return self._data.get(key, default)


if __name__ == "__main__":
    assert slugify("  Hello   World  ") == "hello-world"
    cache = Cache(capacity=1)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") is None and cache.get("b") == 2
    print("smoke OK")
