"""Which channels does the ATD carry? Prints NAMES and COUNTS only -- never data values.

    python possession/atd_channels.py

1. XML metadata: element tags with counts, and the id/name of every channel-like element and of
   every data-specification entry (the order in which fields appear in a raw line).
2. Raw text: how many comma-separated fields the ball part has, as a histogram (a shape, not data).

Open question this answers: is there a ball-alive / ball-owner field beside the ball x,y?
"""

import collections
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import require  # noqa: E402

# Params
NAME_ATTRS = ("id", "name", "channelId", "playerChannelId", "type")
CHANNEL_HINTS = ("channel", "dataspec", "spec", "split", "register")
MAX_ATTR_LEN = 60


def local(tag):
    return tag.rsplit("}", 1)[-1]


def describe_xml(path):
    root = ET.parse(path).getroot()
    tags = collections.Counter(local(e.tag) for e in root.iter())
    print("XML element tags (count):")
    for tag, n in sorted(tags.items()):
        print("  {:40s} {}".format(tag, n))

    print("\nchannel-like elements (tag: identifying attributes):")
    seen = collections.Counter()
    for e in root.iter():
        tag = local(e.tag)
        if not any(h in tag.lower() for h in CHANNEL_HINTS):
            continue
        attrs = {k: v[:MAX_ATTR_LEN] for k, v in e.attrib.items() if k in NAME_ATTRS}
        key = (tag, tuple(sorted(attrs.items())))
        seen[key] += 1
    for (tag, attrs), n in seen.items():
        print("  {:28s} {}  x{}".format(tag, dict(attrs), n))

    print("\nattribute names used, per tag:")
    attr_names = collections.defaultdict(set)
    for e in root.iter():
        attr_names[local(e.tag)].update(e.attrib.keys())
    for tag in sorted(attr_names):
        if attr_names[tag]:
            print("  {:40s} {}".format(tag, sorted(attr_names[tag])))


def describe_raw(path, max_lines=None):
    widths = collections.Counter()
    parts_count = collections.Counter()
    bad = 0
    with open(path) as handle:
        for k, line in enumerate(handle):
            if max_lines and k >= max_lines:
                break
            try:
                parts = line.rstrip("\r\n").split(":")
                parts_count[len(parts)] += 1
                widths[len(parts[-1].split(","))] += 1
            except Exception:                            # never echo the line
                bad += 1
    print("\nraw lines by number of ':' sections: {}".format(dict(parts_count)))
    print("ball section, number of ',' fields: {}".format(dict(widths)))
    if bad:
        print("unparseable lines: {}".format(bad))


def main():
    meta, raw = require(["atd_meta", "atd_raw"])
    describe_xml(meta)
    describe_raw(raw)


if __name__ == "__main__":
    main()
