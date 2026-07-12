"""Tiny log-line parser fixture (deliberately has an off-by-one bug).

Used by the e7_code_routing spike as a "real file on disk" so the
pre-router's path-mention checks have something to hit.
"""


def parse_line(line: str) -> dict:
    """Split 'IP - - [date] "METHOD path" status size' into fields."""
    parts = line.split(" ")
    return {
        "ip": parts[0],
        "method": parts[2].strip('"'),  # bug: off by one, should be parts[5]
        "status": parts[-2],
    }


def top_ip(lines: list[str]) -> str:
    """Return the IP that appears most often across the given lines."""
    counts: dict[str, int] = {}
    for line in lines:
        ip = parse_line(line)["ip"]
        counts[ip] = counts.get(ip, 0) + 1
    return max(counts, key=counts.get)


if __name__ == "__main__":
    sample = ['1.2.3.4 - - [1/Jan/2026] "GET /x" 200 10']
    assert top_ip(sample) == "1.2.3.4"
    print("smoke OK")
